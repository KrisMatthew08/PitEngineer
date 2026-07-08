"""PitEngineer desktop app - a race-engineer dashboard (Tkinter).

Dark, motorsport-styled. Detects your car from the running game, learns the
car's parameters from your setups, and runs the drive -> debrief -> apply loop:
stat tiles (best / median / consistency / gap), where you lose time by corner,
and the proposed setup change as the hero card. Long work (telemetry capture,
AI diagnosis) runs on worker threads so the window stays responsive.

    python -m pitengineer.gui       (or double-click PitEngineer.exe)
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext

from .car_data import (build_manifest_from_setups, find_current_setup,
                       track_setup_target)
from .engines import make_engine
from .ollama_manager import BUNDLED_MODEL, bundled_model_name, ensure_running
from .session_log import SessionMemory, StintRecord
from .setup_file import load_setup, writable_target, write_setup
from .shared_memory import read_car_track, session_status
from .stint import StintRecorder, analyze, fmt_time
from .translator import diagnose_autotune

# --- palette (dark, motorsport) ---
BG = "#14171c"
PANEL = "#1d222b"
PANEL2 = "#232a35"
FG = "#eaedf2"
MUTED = "#8b95a5"
ACCENT = "#e10600"       # racing red
GOOD = "#2ecc71"
BAD = "#e74c3c"
WARN = "#f1c40f"
DOT = {"LIVE": GOOD, "PAUSE": WARN, "OFF": BAD, "REPLAY": MUTED}


class AutoTuneApp:
    def __init__(self, root: tk.Tk, engine_kind: str = "ollama",
                 model: str | None = None) -> None:
        self.root = root
        root.title("PitEngineer — AI Race Engineer")
        root.configure(bg=BG)
        root.minsize(820, 720)

        self.engine = make_engine(engine_kind, model)
        self.memory = SessionMemory()
        self.car = ""
        self.track = ""
        self.manifest = None
        self.setup = None
        self.setup_path = None
        self.recorder: StintRecorder | None = None
        self.last_change: dict[str, tuple[int, int]] | None = None
        self.pending = None
        self.stint_no = 0
        self.recording = False

        self._fonts()
        self._build()
        self._poll_ac()
        # Auto-start the bundled/system Ollama in the background so the user
        # never has to launch it. Analysis + Full Setup Pass work regardless.
        threading.Thread(target=self._start_ai, daemon=True).start()

    def _start_ai(self) -> None:
        self.root.after(0, self._status, "Starting AI engine…")
        ok = ensure_running()
        if ok:
            self.root.after(0, self._status, "AI ready. Start AC, get on track, then Detect car.")
        else:
            self.root.after(0, self._status,
                            "AI engine unavailable — analysis and Full Setup Pass "
                            "still work without it.")

    # ---------- fonts ----------
    def _fonts(self) -> None:
        self.f_title = tkfont.Font(family="Segoe UI Semibold", size=15)
        self.f_sub = tkfont.Font(family="Segoe UI", size=9)
        self.f_tile_val = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.f_tile_cap = tkfont.Font(family="Segoe UI", size=8)
        self.f_h = tkfont.Font(family="Segoe UI Semibold", size=10)
        self.f_body = tkfont.Font(family="Segoe UI", size=10)
        self.f_change = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.f_btn = tkfont.Font(family="Segoe UI Semibold", size=11)

    # ---------- widget helpers ----------
    def _panel(self, parent, **kw):
        return tk.Frame(parent, bg=PANEL, highlightthickness=0, **kw)

    def _btn(self, parent, text, cmd, accent=True):
        b = tk.Button(parent, text=text, command=cmd, font=self.f_btn,
                      bg=ACCENT if accent else PANEL2, fg="white",
                      activebackground="#b40500" if accent else "#2c3542",
                      activeforeground="white", relief="flat", bd=0,
                      padx=18, pady=9, cursor="hand2",
                      disabledforeground=MUTED)
        return b

    # ---------- build ----------
    def _build(self) -> None:
        pad = 14

        # Header
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=pad, pady=(pad, 6))
        self.dot = tk.Label(head, text="●", font=("Segoe UI", 16), bg=BG, fg=BAD)
        self.dot.pack(side="left")
        htext = tk.Frame(head, bg=BG)
        htext.pack(side="left", padx=8)
        self.title_lbl = tk.Label(htext, text="Waiting for Assetto Corsa…",
                                  font=self.f_title, bg=BG, fg=FG, anchor="w")
        self.title_lbl.pack(anchor="w")
        self.sub_lbl = tk.Label(htext, text=f"Engine: {self.engine.name}",
                                font=self.f_sub, bg=BG, fg=MUTED, anchor="w")
        self.sub_lbl.pack(anchor="w")
        self.detect_btn = self._btn(head, "Detect car", self._detect, accent=False)
        self.detect_btn.pack(side="right")

        # Stat tiles
        tiles = tk.Frame(self.root, bg=BG)
        tiles.pack(fill="x", padx=pad, pady=6)
        self.tiles = {}
        for i, (key, cap) in enumerate([("best", "BEST LAP"), ("median", "TYPICAL PACE"),
                                        ("consist", "CONSISTENCY"), ("gap", "vs BEST-EVER")]):
            card = self._panel(tiles)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            tiles.columnconfigure(i, weight=1)
            val = tk.Label(card, text="—", font=self.f_tile_val, bg=PANEL, fg=FG)
            val.pack(pady=(12, 0))
            tk.Label(card, text=cap, font=self.f_tile_cap, bg=PANEL, fg=MUTED).pack(pady=(0, 10))
            self.tiles[key] = val

        # Info panel (balance / tyres / corner)
        info = self._panel(self.root)
        info.pack(fill="x", padx=pad, pady=6)
        self.balance_lbl = tk.Label(info, text="Balance: —", font=self.f_body,
                                    bg=PANEL, fg=FG, anchor="w", justify="left")
        self.balance_lbl.pack(fill="x", padx=12, pady=(10, 2))
        self.tyres_lbl = tk.Label(info, text="Tyres: —", font=self.f_body,
                                  bg=PANEL, fg=FG, anchor="w")
        self.tyres_lbl.pack(fill="x", padx=12, pady=2)
        self.corner_lbl = tk.Label(info, text="Time loss: —", font=self.f_body,
                                   bg=PANEL, fg=WARN, anchor="w", justify="left",
                                   wraplength=760)
        self.corner_lbl.pack(fill="x", padx=12, pady=(2, 10))

        # Hero: proposed change card
        hero = self._panel(self.root)
        hero.pack(fill="x", padx=pad, pady=6)
        bar = tk.Frame(hero, bg=ACCENT, height=3)
        bar.pack(fill="x")
        tk.Label(hero, text="PROPOSED CHANGE", font=self.f_h, bg=PANEL,
                 fg=ACCENT).pack(anchor="w", padx=12, pady=(10, 4))
        self.change_box = tk.Frame(hero, bg=PANEL)
        self.change_box.pack(fill="x", padx=12, pady=(0, 6))
        self.change_placeholder = tk.Label(
            self.change_box, text="Drive a stint and I'll propose a change here.",
            font=self.f_body, bg=PANEL, fg=MUTED, anchor="w")
        self.change_placeholder.pack(anchor="w")
        self.apply_btn = self._btn(hero, "Apply change & continue", self._apply)
        self.apply_btn.configure(state="disabled")
        self.apply_btn.pack(anchor="e", padx=12, pady=(0, 12))

        # Detail log (full debrief text, scrollable)
        logwrap = self._panel(self.root)
        logwrap.pack(fill="both", expand=True, padx=pad, pady=6)
        tk.Label(logwrap, text="DETAILS", font=self.f_h, bg=PANEL,
                 fg=MUTED).pack(anchor="w", padx=12, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(
            logwrap, wrap="word", height=8, bg=PANEL, fg=MUTED, bd=0,
            font=("Consolas", 9), insertbackground=FG, relief="flat")
        self.log.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self.log.configure(state="disabled")

        # Bottom bar
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=pad, pady=(0, 6))
        self.stint_btn = self._btn(bottom, "● Start stint", self._toggle_stint)
        self.stint_btn.configure(state="disabled")
        self.stint_btn.pack(side="left")
        self.full_var = tk.BooleanVar(value=False)
        self.full_chk = tk.Checkbutton(
            bottom, text="Full setup pass (change everything)", variable=self.full_var,
            font=self.f_sub, bg=BG, fg=MUTED, selectcolor=PANEL2,
            activebackground=BG, activeforeground=FG, bd=0, highlightthickness=0,
            cursor="hand2")
        self.full_chk.pack(side="left", padx=12)
        self.busy = tk.Label(bottom, text="", font=self.f_sub, bg=BG, fg=ACCENT)
        self.busy.pack(side="right")

        self.statusbar = tk.Label(self.root, text="Start Assetto Corsa and get on track.",
                                  font=self.f_sub, bg=PANEL2, fg=MUTED, anchor="w")
        self.statusbar.pack(fill="x", side="bottom", ipady=3)

    # ---------- small ops ----------
    def _log(self, text: str, header: bool = False) -> None:
        self.log.configure(state="normal")
        if header:
            self.log.insert("end", "\n" + "─" * 60 + "\n")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _status(self, t: str) -> None:
        self.statusbar.configure(text=t)

    def _busy(self, on: bool, text: str = "working…") -> None:
        self.busy.configure(text=text if on else "")

    def _set_tile(self, key: str, value: str, color: str = FG) -> None:
        self.tiles[key].configure(text=value, fg=color)

    # ---------- AC polling ----------
    def _poll_ac(self) -> None:
        status = session_status()
        self.dot.configure(fg=DOT.get(status, MUTED))
        if self.car and self.manifest:
            self.title_lbl.configure(text=f"{self.manifest.display_name}  ·  {self.track}")
            self.sub_lbl.configure(
                text=f"{self.engine.name}   ·   AC: {status}   ·   "
                     f"{len(self.manifest.parameters)} params   ·   "
                     f"{self.setup_path.name if self.setup_path else '-'}")
        else:
            self.title_lbl.configure(text=f"Assetto Corsa: {status}")
        self.root.after(1500, self._poll_ac)

    # ---------- detect ----------
    def _detect(self) -> None:
        try:
            car, track = read_car_track()
        except (OSError, FileNotFoundError):
            car, track = "", ""
        if not car:
            messagebox.showwarning("No car detected",
                                   "Make sure Assetto Corsa is running and you're "
                                   "on track (not the menu), then retry.")
            return
        try:
            manifest = build_manifest_from_setups(car, display_name=car)
        except FileNotFoundError as exc:
            messagebox.showerror("No setups found", str(exc))
            return
        setup_path = find_current_setup(car, track)
        if not setup_path:
            messagebox.showerror("No setup file",
                                 f"No setup found for {car} / {track}. Save one in-game.")
            return
        switched = bool(self.car) and (car != self.car or track != self.track)
        self.car, self.track = car, track
        self.manifest, self.setup_path = manifest, setup_path
        self.setup = load_setup(setup_path)
        if switched:
            # New car/track: drop state tied to the previous one so we don't
            # judge this car against the old car's last change.
            self.last_change = None
            self.pending = None
            self.stint_no = 0
            self.apply_btn.configure(state="disabled")
        self.stint_btn.configure(state="normal")
        self._log(f"Detected {car} @ {track} — learned {len(manifest.parameters)} "
                  f"params from your setups. Tuning {setup_path.name}.", header=True)
        self._status("Ready. Press Start stint, drive a few laps, then Stop & analyze.")

    # ---------- stint ----------
    def _toggle_stint(self) -> None:
        if not self.recording:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if session_status() not in ("LIVE", "PAUSE"):
            messagebox.showwarning("Not on track", "Get out on track first.")
            return
        self.recorder = StintRecorder()
        self.recorder.start()
        self.recording = True
        self.stint_btn.configure(text="■ Stop & analyze")
        self.apply_btn.configure(state="disabled")
        self._status("Recording… drive your laps, then Stop & analyze.")

    def _stop(self) -> None:
        self.recording = False
        self.stint_btn.configure(state="disabled", text="● Start stint")
        data = self.recorder.stop() if self.recorder else None
        if self.recorder and self.recorder.error:
            messagebox.showerror("Telemetry error", str(self.recorder.error))
            self.stint_btn.configure(state="normal")
            return
        self._status("Analyzing telemetry and diagnosing…")
        self._busy(True, "analyzing…")
        threading.Thread(target=self._worker, args=(data,), daemon=True).start()

    def _worker(self, data) -> None:
        try:
            report = analyze(data)
            record = StintRecord.from_report(self.car, self.track, report, self.last_change)
            prev = self.memory.last(self.car, self.track)
            verdict = self.memory.compare(prev, record)
            self.memory.append(record)
            from . import segments
            ghost = self.memory.load_ghost(self.car, self.track)
            seg = segments.analyze(data, reference=ghost)
            if seg.lap_time_s > 0 and (ghost is None or seg.lap_time_s < ghost.get("lap_time_s", 1e9)):
                self.memory.save_ghost(self.car, self.track, segments.to_reference(seg))
            diag = diagnose_autotune(report, verdict if self.last_change else None,
                                     self.setup, self.manifest, self.engine,
                                     self.last_change, segment_context=seg.worst_summary(),
                                     full_pass=self.full_var.get())
            self.root.after(0, self._show, report, verdict, seg, diag)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._error, exc)

    def _error(self, exc: Exception) -> None:
        self._busy(False)
        self.stint_btn.configure(state="normal")
        messagebox.showerror("Diagnosis failed", str(exc))
        self._status("Error — try another stint.")

    # ---------- render results ----------
    def _show(self, report, verdict, seg, diag) -> None:
        self._busy(False)
        self.stint_no += 1
        m = report.metrics

        # Tiles
        self._set_tile("best", fmt_time(m.best_lap_ms))
        self._set_tile("median", fmt_time(m.median_lap_ms))
        c = report.profile.consistency
        self._set_tile("consist", f"{c:.2f}", GOOD if c > 0.66 else WARN if c > 0.33 else BAD)
        if seg.reference_gap_s is not None:
            g = seg.reference_gap_s
            self._set_tile("gap", f"{g:+.2f}s", GOOD if g < 0 else BAD if g > 0 else FG)
        else:
            self._set_tile("gap", "baseline", MUTED)

        # Info
        s = report.summary
        self.balance_lbl.configure(
            text=f"Balance: {s.tendency_strength} {s.tendency}   ·   "
                 f"Gearing: {report.gearing.issue.replace('_',' ') or 'ok'}")
        fl, fr, rl, rr = s.tyre_temp
        pr = report.pressures
        self.tyres_lbl.configure(
            text=f"Tyre temps °C   FL {fl:.0f} · FR {fr:.0f} · RL {rl:.0f} · RR {rr:.0f}"
                 f"      ·      Hot psi  F {pr.front_psi:.1f} · R {pr.rear_psi:.1f}")
        loss = seg.worst_summary() or (
            report.consistency_note() or "Car looks balanced — good baseline.")
        self.corner_lbl.configure(text=loss)

        # Hero: proposed change
        for w in self.change_box.winfo_children():
            w.destroy()
        if diag.changes:
            for c_ in diag.changes:
                row = tk.Frame(self.change_box, bg=PANEL)
                row.pack(fill="x", pady=2)
                arrow = (f"{c_.label}   {c_.human_current(self.manifest)} → "
                         f"{c_.human_proposed(self.manifest)}")
                tk.Label(row, text=arrow, font=self.f_change, bg=PANEL, fg=FG,
                         anchor="w").pack(anchor="w")
                tk.Label(row, text=f"   {c_.reason}  ({c_.confidence})",
                         font=self.f_sub, bg=PANEL, fg=MUTED, anchor="w",
                         wraplength=740, justify="left").pack(anchor="w")
            self.pending = diag
            self.apply_btn.configure(state="normal")
            self._status("Review the proposed change, then Apply & continue.")
        else:
            from .translator import _clear_problem
            msg = (">> Dialled in for your driving." if _clear_problem(report, self.manifest) is None
                   else ">> Couldn't auto-fix this on this car — adjust manually, or use the Claude engine.")
            tk.Label(self.change_box, text=msg, font=self.f_body, bg=PANEL,
                     fg=MUTED, anchor="w").pack(anchor="w")
            self.pending = None
            self._status("No change this stint — see details.")

        # Detail log
        self._log(f"STINT {self.stint_no}", header=True)
        self._log(report.describe())
        if seg.segments:
            self._log("\n" + seg.describe())
        if self.last_change:
            self._log("\n" + verdict.text)
        prog = self.memory.progress(self.car, self.track)
        if prog:
            self._log(prog)
        self.stint_btn.configure(state="normal")

    # ---------- apply ----------
    def _apply(self) -> None:
        if not self.pending:
            return
        changes = {c.section: c.proposed_index for c in self.pending.changes}
        # Save into the LIVE track's folder (…/<car>/<track>/pitengineer.ini) so
        # AC loads it for this track - not generic. Fall back to redirecting
        # last.ini -> pitengineer.ini in the source folder if the track is unknown.
        target = track_setup_target(self.car, self.track) or writable_target(self.setup_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        out = None if target == self.setup_path else target
        written = write_setup(self.setup, changes, out_path=out, backup=True)
        self.last_change = {c.section: (c.current_index, c.proposed_index)
                            for c in self.pending.changes}
        # From now on, work on the file AC will actually load.
        self.setup_path = written
        self.setup = load_setup(written)
        self.pending = None
        self.apply_btn.configure(state="disabled")
        for w in self.change_box.winfo_children():
            w.destroy()
        name = written.stem
        tk.Label(self.change_box, text=f"✓ Applied. In the pits, open Setup and "
                 f"LOAD the '{name}' setup (no game restart needed), then drive "
                 "the next stint.", font=self.f_body, bg=PANEL, fg=GOOD,
                 anchor="w", wraplength=740, justify="left").pack(anchor="w")
        self._status(f"Applied to '{name}'. Load it in the pits, then Start stint.")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="PitEngineer (GUI)")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # Use whatever model is actually bundled; fall back to the default in dev.
    model = args.model
    if model is None and args.engine == "ollama":
        model = bundled_model_name() or BUNDLED_MODEL
    root = tk.Tk()
    AutoTuneApp(root, args.engine, model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
