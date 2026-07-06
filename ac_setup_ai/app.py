"""Command-line Translator: describe the problem, get validated changes, apply.

Usage:
    python -m ac_setup_ai.app --setup data/sample_setup.ini \\
        --manifest data/manifests/generic_gt3.json

Then type how the car feels (e.g. "rear steps out on power exit"). The app
prints a diagnosis + before/after diff and asks before writing the .ini.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .engines import Engine, make_engine
from .manifest import CarManifest, load_manifest
from .setup_file import Setup, load_setup, write_setup
from .translator import Diagnosis, diagnose


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines into os.environ.

    Only sets variables that aren't already present in the environment.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _print_diagnosis(diag: Diagnosis, manifest: CarManifest) -> None:
    print()
    print("Diagnosis:")
    print(f"  {diag.text}")
    print()
    if not diag.changes:
        print("No changes proposed (the model found nothing worth changing).")
        return

    print("Proposed changes (most impactful first):")
    print("-" * 68)
    for i, c in enumerate(diag.changes, 1):
        arrow = f"{c.human_current(manifest)}  ->  {c.human_proposed(manifest)}"
        print(f"{i}. {c.label}  [{c.section}]")
        print(f"   {arrow}    (confidence: {c.confidence})")
        print(f"   why: {c.reason}")
        if c.clamped:
            print("   note: value was clamped into the car's legal range.")
        print()


def _apply(diag: Diagnosis, setup: Setup) -> None:
    changes = {c.section: c.proposed_index for c in diag.changes}
    written = write_setup(setup, changes, backup=True)
    print(f"\nApplied {len(changes)} change(s). Wrote: {written}")
    print(f"Backup of the original saved next to it as {written.name}.bak")
    print("In-game: re-select / reload this setup in the pits for it to take effect.")


def run(setup_path: str, manifest_path: str, engine: Engine) -> int:
    setup = load_setup(setup_path)
    manifest = load_manifest(manifest_path)

    print(f"Car:    {manifest.display_name}")
    print(f"Setup:  {Path(setup_path).name}")
    print(f"Engine: {engine.name}")
    print("Describe how the car feels wrong (blank line to quit).")

    while True:
        try:
            complaint = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not complaint:
            return 0

        try:
            diag = diagnose(complaint, setup, manifest, engine)
        except Exception as exc:  # noqa: BLE001 - surface any API/parse failure plainly
            print(f"\nCould not get a diagnosis: {exc}", file=sys.stderr)
            continue

        _print_diagnosis(diag, manifest)
        if not diag.changes:
            continue

        answer = input("Apply these changes to the setup file? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            _apply(diag, setup)
            # Reload so subsequent complaints see the new baseline.
            setup = load_setup(setup_path)
        else:
            print("Left the setup unchanged.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assetto Corsa Setup AI — Translator")
    parser.add_argument("--setup", required=True, help="Path to a setup .ini file")
    parser.add_argument("--manifest", required=True, help="Path to the car manifest JSON")
    parser.add_argument(
        "--engine", default="ollama", choices=["ollama", "claude"],
        help="AI backend: 'ollama' (local, free, default) or 'claude' (needs API key)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model override (e.g. qwen2.5:7b for Ollama, claude-opus-4-8 for Claude)",
    )
    args = parser.parse_args()
    _load_dotenv()
    engine = make_engine(args.engine, args.model)
    return run(args.setup, args.manifest, engine)


if __name__ == "__main__":
    raise SystemExit(main())
