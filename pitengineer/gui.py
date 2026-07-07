"""Native desktop app (Tkinter) for the auto-tune loop.

A real window, no terminal: it detects your car from the running game, learns
the car's parameters from your setups, and runs the drive -> debrief -> apply
loop with buttons. Long work (telemetry capture, AI diagnosis) runs on worker
threads so the window stays responsive.

    python -m pitengineer.gui

Packaged to a double-click .exe via PyInstaller (see build_exe.py / README).
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from .car_data import build_manifest_from_setups, find_current_setup
from .engines import make_engine
from .session_log import SessionMemory, StintRecord
from .setup_file import load_setup, write_setup
from .shared_memory import read_car_track, session_status
from .stint import StintRecorder, analyze
from .translator import diagnose_autotune

_STATUS_COLOR = {"LIVE": "#2e8b57", "PAUSE": "#c8a200", "OFF": "#b03030",
                 "REPLAY": "#888888"}


class AutoTuneApp:
    def __init__(self, root: tk.Tk, engine_kind: str = "ollama",
                 model: str | None = None) -> None:
        self.root = root
        root.title("PitEngineer — AI Race Engineer for Assetto Corsa")
        root.minsize(720, 560)

        self.engine = make_engine(engine_kind, model)
        self.memory = SessionMemory()
        self.car = ""
        self.track = ""
        self.manifest = None
        self.setup = None
        self.setup_path: Path | None = None
        self.recorder: StintRecorder | None = None
        self.last_change: dict[str, tuple[int, int]] | None = None
        self.pending = None          # Diagnosis awaiting apply
        self.stint_no = 0
        self.recording = False

        self._build_ui()
        self._poll_ac()

    # ---------- UI construction ----------
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        self.status_dot = tk.Label(top, text="●", font=("Segoe UI", 14))
        self.status_dot.grid(row=0, column=0, sticky="w")
        self.header = ttk.Label(top, text="Waiting for Assetto Corsa...",
                                font=("Segoe UI", 12, "bold"))
        self.header.grid(row=0, column=1, sticky="w", padx=6)
        self.sub = ttk.Label(top, text=f"Engine: {self.engine.name}",
                             foreground="#666")
        self.sub.grid(row=1, column=1, sticky="w", padx=6)
        self.detect_btn = ttk.Button(top, text="Detect car", command=self._detect)
        self.detect_btn.grid(row=0, column=2, rowspan=2, sticky="e")
        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10)
        self.output = scrolledtext.ScrolledText(mid, wrap="word", height=20,
                                                font=("Consolas", 10))
        self.output.pack(fill="both", expand=True)
        self.output.configure(state="disabled")
        self._log("Start Assetto Corsa, get on track, then press 'Detect car'.\n"
                  "Once detected: press 'Start stint', drive a few laps, then "
                  "'Stop & analyze'. I'll read your telemetry and propose a "
                  "setup change tuned to how you drive.")

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", **pad)
        self.stint_btn = ttk.Button(bottom, text="Start stint",
                                    command=self._toggle_stint, state="disabled")
        self.stint_btn.pack(side="left")
        self.apply_btn = ttk.Button(bottom, text="Apply change & continue",
                                    command=self._apply, state="disabled")
        self.apply_btn.pack(side="left", padx=8)
        self.busy = ttk.Progressbar(bottom, mode="indeterminate", length=160)
        self.busy.pack(side="right")
        self.statusbar = ttk.Label(self.root, text="Ready", relief="sunken",
                                   anchor="w")
        self.statusbar.pack(fill="x", side="bottom")

    # ---------- helpers ----------
    def _log(self, text: str, header: bool = False) -> None:
        self.output.configure(state="normal")
        if header:
            self.output.insert("end", "\n" + "=" * 58 + "\n")
        self.output.insert("end", text + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.statusbar.configure(text=text)

    def _set_busy(self, on: bool) -> None:
        if on:
            self.busy.start(12)
        else:
            self.busy.stop()

    # ---------- AC polling ----------
    def _poll_ac(self) -> None:
        status = session_status()
        self.status_dot.configure(foreground=_STATUS_COLOR.get(status, "#888"))
        if self.car:
            self.header.configure(text=f"{self.manifest.display_name}  @  {self.track}")
            self.sub.configure(
                text=f"Engine: {self.engine.name}   |   AC: {status}   |   "
                     f"{len(self.manifest.parameters)} params   |   "
                     f"setup: {self.setup_path.name if self.setup_path else '-'}"
            )
        else:
            self.header.configure(text=f"Assetto Corsa: {status}")
        self.root.after(1500, self._poll_ac)

    # ---------- detect ----------
    def _detect(self) -> None:
        try:
            car, track = read_car_track()
        except (OSError, FileNotFoundError):
            car, track = "", ""
        if not car:
            messagebox.showwarning(
                "No car detected",
                "Couldn't read a car from Assetto Corsa. Make sure AC is "
                "running and you're on track (not the main menu), then retry.")
            return
        try:
            manifest = build_manifest_from_setups(car, display_name=car)
        except FileNotFoundError as exc:
            messagebox.showerror("No setups found", str(exc))
            return
        setup_path = find_current_setup(car, track)
        if not setup_path:
            messagebox.showerror(
                "No setup file",
                f"No setup found for {car} / {track}. Save a setup in-game first.")
            return

        self.car, self.track = car, track
        self.manifest = manifest
        self.setup_path = setup_path
        self.setup = load_setup(setup_path)
        self.stint_btn.configure(state="normal")
        self._log(f"Detected: {car} @ {track}\n"
                  f"Learned {len(manifest.parameters)} adjustable parameters from "
                  f"your setups.\nTuning setup file: {setup_path.name}", header=True)
        self._set_status("Ready to record a stint.")

    # ---------- stint recording ----------
    def _toggle_stint(self) -> None:
        if not self.recording:
            self._start_recording()
        else:
            self._stop_and_analyze()

    def _start_recording(self) -> None:
        if session_status() not in ("LIVE", "PAUSE"):
            messagebox.showwarning("Not on track",
                                   "Get out on track (a LIVE session) first.")
            return
        self.recorder = StintRecorder()
        self.recorder.start()
        self.recording = True
        self.stint_btn.configure(text="Stop & analyze")
        self.apply_btn.configure(state="disabled")
        self._set_status("Recording... drive your laps, then Stop & analyze.")

    def _stop_and_analyze(self) -> None:
        self.recording = False
        self.stint_btn.configure(state="disabled", text="Start stint")
        data = self.recorder.stop() if self.recorder else None
        if self.recorder and self.recorder.error:
            messagebox.showerror("Telemetry error", str(self.recorder.error))
            self.stint_btn.configure(state="normal")
            return
        self._set_status("Analyzing telemetry and diagnosing (this can take a bit)...")
        self._set_busy(True)
        threading.Thread(target=self._analyze_worker, args=(data,), daemon=True).start()

    def _analyze_worker(self, data) -> None:
        """Runs off the UI thread: analyse + AI diagnosis, then post back."""
        try:
            report = analyze(data)
            record = StintRecord.from_report(self.car, self.track, report, self.last_change)
            prev = self.memory.last(self.car, self.track)
            verdict = self.memory.compare(prev, record)
            self.memory.append(record)
            diag = diagnose_autotune(report, verdict if self.last_change else None,
                                     self.setup, self.manifest, self.engine,
                                     self.last_change)
            self.root.after(0, self._on_diagnosis, report, verdict, diag)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_error, exc)

    def _on_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self.stint_btn.configure(state="normal")
        messagebox.showerror("Diagnosis failed", str(exc))
        self._set_status("Error - try another stint.")

    def _on_diagnosis(self, report, verdict, diag) -> None:
        self._set_busy(False)
        self.stint_no += 1
        self._log(f"STINT {self.stint_no} DEBRIEF", header=True)
        self._log(report.describe())
        if self.last_change:
            self._log("\n" + verdict.text)
        prog = self.memory.progress(self.car, self.track)
        if prog:
            self._log(prog)

        if not diag.changes:
            from .translator import _clear_problem
            self._log(f"\n>>> {diag.text}")
            if _clear_problem(report, self.manifest) is None:
                self._log(">>> The car looks dialled in for your driving.")
                self._set_status("Dialled in. Drive more stints to keep refining if you like.")
            else:
                self._log(">>> Couldn't auto-fix this one on this car - adjust it "
                          "manually, or run with the Claude engine for sharper reasoning.")
                self._set_status("No safe auto-fix for this one - see the note above.")
            self.pending = None
            self.stint_btn.configure(state="normal")
            return

        self._log("\nProposed changes (most impactful first):")
        for i, c in enumerate(diag.changes, 1):
            arrow = f"{c.human_current(self.manifest)} -> {c.human_proposed(self.manifest)}"
            self._log(f"  {i}. {c.label} [{c.section}]  {arrow}  ({c.confidence})")
            self._log(f"     why: {c.reason}")
        self.pending = diag
        self.apply_btn.configure(state="normal")
        self.stint_btn.configure(state="normal")
        self._set_status("Review the proposed change, then Apply & continue.")

    # ---------- apply ----------
    def _apply(self) -> None:
        if not self.pending:
            return
        changes = {c.section: c.proposed_index for c in self.pending.changes}
        written = write_setup(self.setup, changes, backup=True)
        self.last_change = {c.section: (c.current_index, c.proposed_index)
                            for c in self.pending.changes}
        self.setup = load_setup(self.setup_path)
        self.pending = None
        self.apply_btn.configure(state="disabled")
        self._log(f"\nApplied {len(changes)} change(s) to {written.name} "
                  f"(backup: {written.name}.bak).")
        self._log(">>> RELOAD the setup in the pits (re-enter garage / re-select "
                  "setup) so AC applies it, then drive the next stint.")
        self._set_status("Applied. Reload in pits, then Start stint again.")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AC Setup AI (GUI)")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")  # native-ish on Windows
    except tk.TclError:
        pass
    AutoTuneApp(root, args.engine, args.model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
