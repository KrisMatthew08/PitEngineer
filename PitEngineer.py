"""Double-click / packaging entry point for the PitEngineer desktop app.

Run directly:      python PitEngineer.py
Packaged to .exe:  see build_exe.py  ->  dist/PitEngineer.exe
"""

from pitengineer.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
