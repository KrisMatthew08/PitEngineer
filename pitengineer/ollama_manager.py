"""Find and auto-start Ollama so the user never has to install/launch it.

For the self-contained build, a copy of `ollama.exe` and a model are bundled
inside the PitEngineer executable (PyInstaller unpacks them to a temp dir). This
module finds that bundled Ollama - or a system install as a fallback - and
starts `ollama serve` in the background, pointed at the bundled model store, so
the existing OllamaEngine (which talks to localhost:11434) just works offline.

If Ollama is already running, it does nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which

HOST = "http://localhost:11434"

# Small model shipped in the self-contained build (must match the bundled store).
BUNDLED_MODEL = "llama3.2:3b"


def _resource_dir() -> Path:
    """Where bundled resources live (PyInstaller temp dir when frozen)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def find_ollama_exe() -> Path | None:
    """Bundled ollama.exe first, then a system install, then PATH."""
    candidates = [
        _resource_dir() / "ollama" / "ollama.exe",   # bundled with PitEngineer
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    found = which("ollama") or which("ollama.exe")
    return Path(found) if found else None


def bundled_models_dir() -> Path | None:
    """A model store bundled with the exe, if present."""
    d = _resource_dir() / "ollama_models"
    return d if d.exists() else None


def bundled_model_name() -> str | None:
    """Which model is actually bundled (scans the bundled store), so the app
    uses whatever was packaged rather than a hardcoded guess. None in dev."""
    d = bundled_models_dir()
    if d is None:
        return None
    lib = d / "manifests" / "registry.ollama.ai" / "library"
    if not lib.exists():
        return None
    for name_dir in sorted(lib.iterdir()):
        for tag in sorted(name_dir.iterdir()):
            if tag.is_file():
                return f"{name_dir.name}:{tag.name}"
    return None


def is_running(host: str = HOST) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_running(timeout: float = 40.0) -> bool:
    """Make sure an Ollama server is up. Starts the bundled/system one if not.

    Returns True if Ollama is reachable (already-running or successfully started).
    """
    if is_running():
        return True

    exe = find_ollama_exe()
    if exe is None:
        return False

    env = dict(os.environ)
    mdir = bundled_models_dir()
    if mdir is not None:
        env["OLLAMA_MODELS"] = str(mdir)

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        subprocess.Popen(
            [str(exe), "serve"],
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if is_running():
            return True
        time.sleep(0.5)
    return False


def has_model(model: str, host: str = HOST) -> bool:
    """Whether Ollama already has the given model pulled/available."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            import json
            names = [m.get("name", "") for m in json.loads(resp.read()).get("models", [])]
        base = model.split(":")[0]
        return any(n == model or n.split(":")[0] == base for n in names)
    except (urllib.error.URLError, OSError, ValueError):
        return False
