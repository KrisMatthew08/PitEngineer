"""Auto-tune: drive stints, and the app converges your setup to your style.

The full loop, no typing:

    1. It detects your car + track from the running game.
    2. It learns the car's parameters from your own setups.
    3. You drive a stint (a few laps). It reads your telemetry.
    4. It debriefs: what the car did, how you drive, did the last change help.
    5. It proposes the next change (tailored to you). You press A to apply.
    6. Reload the setup in the pits, drive again. It refines.
    7. When gains plateau and the car is balanced, it says "dialled in".

    python -m pitengineer.autotune            # fully auto-detected
    python -m pitengineer.autotune --setup <file> --car <id>   # overrides

Needs Assetto Corsa running and you on track.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import _load_dotenv, _print_diagnosis
from .car_data import build_manifest_from_setups, find_current_setup
from .engines import Engine, make_engine
from .manifest import CarManifest
from .session_log import SessionMemory, StintRecord
from .setup_file import Setup, load_setup, write_setup
from .shared_memory import read_car_track, session_status
from .stint import StintRecorder, analyze


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def _capture_stint() -> "StintRecorder | None":
    """Record one stint: start, wait for the driver to finish, stop."""
    if session_status() not in ("LIVE", "PAUSE"):
        print("You're not on track (need a LIVE session). Get out on track first.")
        return None
    rec = StintRecorder()
    rec.start()
    print("  ...recording. Drive your laps, then press Enter to finish the stint.")
    _prompt("")
    rec.stop()
    if rec.error:
        print(f"  telemetry error during capture: {rec.error}", file=sys.stderr)
        return None
    return rec


def run(setup_path: str, manifest: CarManifest, engine: Engine,
        car: str, track: str) -> int:
    setup = load_setup(setup_path)
    memory = SessionMemory()

    print("=" * 62)
    print(f" AUTO-TUNE  |  {manifest.display_name}  @  {track}")
    print(f" Setup:  {Path(setup_path).name}")
    print(f" Engine: {engine.name}")
    print(f" Params: {len(manifest.parameters)} adjustable (learned from your setups)")
    print("=" * 62)

    last_change: dict[str, tuple[int, int]] | None = None
    stint_no = 0

    while True:
        cmd = _prompt(f"\n[Stint {stint_no + 1}] Press Enter to start recording "
                      "(or 'q' to quit): ")
        if cmd == "q":
            print("\nAuto-tune session ended. Your setup and history are saved.")
            return 0

        rec = _capture_stint()
        if rec is None:
            continue
        stint_no += 1

        report = analyze(rec.data)
        record = StintRecord.from_report(car, track, report, last_change)
        prev = memory.last(car, track)
        verdict = memory.compare(prev, record)
        memory.append(record)

        print("\n" + "-" * 62)
        print(f"STINT {stint_no} DEBRIEF")
        print("-" * 62)
        print(report.describe())
        if last_change:
            print("\n" + verdict.text)
        prog = memory.progress(car, track)
        if prog:
            print(prog)

        print("\nWorking out the next step (this can take ~30-60s)...")
        try:
            from .translator import diagnose_autotune
            diag = diagnose_autotune(report, verdict if last_change else None,
                                     setup, manifest, engine, last_change)
        except Exception as exc:  # noqa: BLE001
            print(f"Diagnosis failed: {exc}", file=sys.stderr)
            last_change = None
            continue

        if not diag.changes:
            from .translator import _clear_problem
            print(f"\n>>> {diag.text}")
            if _clear_problem(report, manifest) is None:
                print(">>> The car looks dialled in for your driving. Nice.")
            else:
                print(">>> Couldn't auto-fix this one on this car - adjust it "
                      "manually, or try --engine claude for sharper reasoning.")
            cont = _prompt("Keep tuning anyway? [y/N] ").lower()
            if cont not in ("y", "yes"):
                print("\nDone. Setup and history saved.")
                return 0
            last_change = None
            continue

        _print_diagnosis(diag, manifest)
        ans = _prompt("Apply these changes and continue? [Y/n] ").lower()
        if ans in ("", "y", "yes"):
            changes = {c.section: c.proposed_index for c in diag.changes}
            written = write_setup(setup, changes, backup=True)
            last_change = {c.section: (c.current_index, c.proposed_index) for c in diag.changes}
            setup = load_setup(setup_path)  # reload new baseline
            print(f"\nApplied {len(changes)} change(s) to {written.name} "
                  f"(backup saved as {written.name}.bak).")
            print(">>> Now RELOAD the setup in the pits (re-enter the garage / "
                  "re-select the setup) so AC picks up the changes, then drive "
                  "the next stint.")
        else:
            print("Left the setup unchanged.")
            last_change = None


def _resolve_car_track(args) -> tuple[str, str]:
    car, track = args.car or "", args.track or ""
    if not car or not track:
        try:
            det_car, det_track = read_car_track()
        except (OSError, FileNotFoundError):
            det_car, det_track = "", ""
        car = car or det_car
        track = track or det_track
    return car, track


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assetto Corsa Setup AI — auto-tune to your driving"
    )
    parser.add_argument("--setup", default=None, help="Setup .ini (auto-detected if omitted)")
    parser.add_argument("--car", default=None, help="Car id (auto-detected from AC if omitted)")
    parser.add_argument("--track", default=None, help="Track id (auto-detected from AC if omitted)")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default=None, help="Model override")
    args = parser.parse_args()
    _load_dotenv()

    status = session_status()
    if status == "OFF" and not (args.car and args.setup):
        print(
            "Assetto Corsa isn't running (or its telemetry isn't available).\n"
            "Start AC and get on track, or pass --car and --setup manually.",
            file=sys.stderr,
        )
        return 1

    car, track = _resolve_car_track(args)
    if not car:
        print("Could not determine the car. Pass --car <id>.", file=sys.stderr)
        return 1

    try:
        manifest = build_manifest_from_setups(car, display_name=car)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    setup_path = args.setup or (find_current_setup(car, track) or "")
    if not setup_path or not Path(setup_path).exists():
        print(
            f"Could not find a setup file for {car} / {track}. "
            "Save a setup in-game first, or pass --setup <file>.",
            file=sys.stderr,
        )
        return 1

    engine = make_engine(args.engine, args.model)
    return run(str(setup_path), manifest, engine, car, track)


if __name__ == "__main__":
    raise SystemExit(main())
