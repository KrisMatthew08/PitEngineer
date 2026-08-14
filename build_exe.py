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

import os
import shutil
import subprocess
import sys
from pathlib import Path


DESKTOP_TARGET = Path.home() / "Desktop" / "PitEngineer.exe"


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
      result = subprocess.call(cmd)
      if result != 0:
        return result

      dist = Path("dist") / "PitEngineer.exe"
      if not dist.exists():
        raise SystemExit(f"Build finished but executable was not found: {dist}")

      if DESKTOP_TARGET.exists():
        if DESKTOP_TARGET.is_dir():
          shutil.rmtree(DESKTOP_TARGET)
        else:
          DESKTOP_TARGET.unlink()
      shutil.copy2(dist, DESKTOP_TARGET)
      if os.name == "nt":
        print(f"Copied build to: {DESKTOP_TARGET}")
      return 0


if __name__ == "__main__":
    raise SystemExit(main())
