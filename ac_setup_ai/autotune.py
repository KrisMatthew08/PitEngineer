"""Auto-tune: drive, and the app reads your telemetry and tunes the setup.

No typing. You drive a few laps, the app captures your telemetry, works out
what the car is doing wrong from the data, proposes setup changes, and (on your
OK) writes them to the .ini.

    python -m ac_setup_ai.autotune --setup <your_setup.ini> --manifest <car.json>

Requires Assetto Corsa running and you on track (a LIVE session).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import _apply, _load_dotenv, _print_diagnosis
from .engines import Engine, make_engine
from .manifest import load_manifest
from .setup_file import load_setup
from .shared_memory import ACTelemetry, session_status
from .summarizer import capture, summarize
from .translator import diagnose_from_telemetry


def run(setup_path: str, manifest_path: str, engine: Engine, seconds: float) -> int:
    setup = load_setup(setup_path)
    manifest = load_manifest(manifest_path)

    print(f"Car:    {manifest.display_name}")
    print(f"Setup:  {Path(setup_path).name}")
    print(f"Engine: {engine.name}")

    status = session_status()
    if status == "OFF":
        print(
            "\nAssetto Corsa isn't running (or no session). Start AC, get on "
            "track, then run this again.",
            file=sys.stderr,
        )
        return 1
    print(f"AC session status: {status}")

    while True:
        print(
            f"\nDrive normally for the next ~{seconds:.0f}s so I can read your "
            "telemetry.\nPress Enter when you're on track and ready (or 'q' to quit)."
        )
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if cmd in ("q", "quit", "exit"):
            return 0

        if session_status() not in ("LIVE", "PAUSE"):
            print("You don't seem to be on track (need a LIVE session). Try again.")
            continue

        print(f"Capturing telemetry for {seconds:.0f}s — keep driving...")
        try:
            with ACTelemetry() as tele:
                samples = capture(duration_s=seconds, tele=tele)
        except (OSError, FileNotFoundError) as exc:
            print(f"Lost the telemetry connection: {exc}", file=sys.stderr)
            continue

        summary = summarize(samples)
        print("\n--- What the telemetry shows ---")
        print(summary.describe())

        if summary.driving_samples < 20:
            print(
                "\nToo few driving samples captured (were you stationary or in "
                "the pits?). Try again while actually lapping."
            )
            continue

        print("\nDiagnosing from your driving (this can take a minute)...")
        try:
            diag = diagnose_from_telemetry(summary, setup, manifest, engine)
        except Exception as exc:  # noqa: BLE001
            print(f"Diagnosis failed: {exc}", file=sys.stderr)
            continue

        _print_diagnosis(diag, manifest)
        if not diag.changes:
            continue

        answer = input("Apply these changes to the setup file? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            _apply(diag, setup)
            setup = load_setup(setup_path)  # reload new baseline for the next run
        else:
            print("Left the setup unchanged.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assetto Corsa Setup AI — auto-tune from live telemetry"
    )
    parser.add_argument("--setup", required=True, help="Path to a setup .ini file")
    parser.add_argument("--manifest", required=True, help="Path to the car manifest JSON")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "claude"])
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument(
        "--seconds", type=float, default=30.0,
        help="How long to capture telemetry each pass (default 30s)",
    )
    args = parser.parse_args()
    _load_dotenv()
    engine = make_engine(args.engine, args.model)
    return run(args.setup, args.manifest, engine, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
