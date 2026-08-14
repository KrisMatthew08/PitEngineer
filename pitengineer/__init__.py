"""Assetto Corsa Setup AI — the Translator MVP.

Describe what the car feels wrong in plain language; get back validated,
in-range setup changes with explanations, and apply them to the .ini.
"""

import subprocess
from pathlib import Path

__version__ = "1.3.0"

# Build timestamp — shows the date of the latest git commit, so you can
# confirm the running app matches what's on GitHub.
def _build_date() -> str:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # "2026-08-14 13:21:13 +0800" -> "2026-08-14 13:21"
        parts = out.split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1][:5]}"
    except Exception:
        pass
    # Fallback: use the mtime of this file
    try:
        import datetime
        mtime = Path(__file__).stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "unknown"

BUILD_DATE = _build_date()
