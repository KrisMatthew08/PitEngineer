"""Build PitEngineer into a single double-click Windows .exe.

Usage:
    pip install pyinstaller
    python build_exe.py

Produces: dist/PitEngineer.exe  (no console window; the GUI is the app)

Notes:
- No data files need bundling: the app learns each car's parameters from your
  Assetto Corsa setups folder at runtime, and talks to Ollama over localhost.
- Antivirus/SmartScreen may flag a fresh unsigned PyInstaller exe the first time;
  that's normal for unsigned binaries.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",            # no console window for the GUI
        "--name", "PitEngineer",
        "PitEngineer.py",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
