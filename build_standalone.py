"""Build the SELF-CONTAINED PitEngineer: Ollama + a model bundled in.

Produces a folder (onedir) the user unzips and runs - no Python, no Ollama
install, no internet, no API key. PitEngineer auto-starts the bundled Ollama
pointed at the bundled model store.

Why onedir (a folder) not onefile: the model is ~1-2GB; a onefile exe would
extract all of it to temp on every launch (slow). A onedir folder starts fast.
Zip the resulting `dist/PitEngineer/` to share it.

Usage:
    pip install pyinstaller
    python build_standalone.py                 # bundles ollama_manager.BUNDLED_MODEL
    python build_standalone.py --model gemma3:1b   # smaller bundle for testing

Requires: Ollama installed with the model already pulled (this script copies
only that model's blobs, not your whole model store).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pitengineer.ollama_manager import BUNDLED_MODEL, find_ollama_exe

STAGING = Path("build_staging")


def ollama_models_root() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    if env and Path(env).exists():
        return Path(env)
    return Path.home() / ".ollama" / "models"


def stage_ollama() -> Path:
    exe = find_ollama_exe()
    if not exe or not exe.exists():
        raise SystemExit("Could not find ollama.exe - install Ollama first.")
    src_root = exe.parent
    dst = STAGING / "ollama"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, dst / "ollama.exe")   # the server; skip the tray app + uninstaller

    # Runtime lives in lib/ollama. Keep the CPU runners + core libs (~35MB) and
    # the small Vulkan backend (broad GPU support), but SKIP the giant
    # cuda_v*/rocm_* folders (~2.9GB, NVIDIA/AMD-specific) - CPU/Vulkan works
    # everywhere and keeps the bundle lean.
    src_lib = src_root / "lib" / "ollama"
    if src_lib.exists():
        dst_lib = dst / "lib" / "ollama"
        dst_lib.mkdir(parents=True, exist_ok=True)
        for f in src_lib.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_lib / f.name)
        vk = src_lib / "vulkan"
        if vk.exists():
            shutil.copytree(vk, dst_lib / "vulkan", dirs_exist_ok=True)
    total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"  staged Ollama runtime ({total/1e6:.0f} MB, CPU+Vulkan, no CUDA/ROCm)")
    return dst


def stage_model(model: str) -> None:
    """Copy ONLY the given model's manifest + referenced blobs."""
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    root = ollama_models_root()
    man_path = (root / "manifests" / "registry.ollama.ai" / "library" / name / tag)
    if not man_path.exists():
        raise SystemExit(f"Model manifest not found: {man_path}\n"
                         f"Pull it first:  ollama pull {model}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    digests = [manifest["config"]["digest"]] + [ly["digest"] for ly in manifest["layers"]]

    dst_man = STAGING / "ollama_models" / "manifests" / "registry.ollama.ai" / "library" / name / tag
    dst_man.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(man_path, dst_man)

    blobs_src = root / "blobs"
    blobs_dst = STAGING / "ollama_models" / "blobs"
    blobs_dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for d in digests:
        fn = d.replace(":", "-")
        src = blobs_src / fn
        if src.exists():
            shutil.copy2(src, blobs_dst / fn)
            total += src.stat().st_size
    print(f"  staged model {model}: {len(digests)} blobs, {total/1e9:.2f} GB")


def build() -> int:
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
        "--windowed", "--name", "PitEngineer",
        "--add-data", f"{STAGING/'ollama'}{sep}ollama",
        "--add-data", f"{STAGING/'ollama_models'}{sep}ollama_models",
        "PitEngineer.py",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BUNDLED_MODEL,
                    help=f"Model to bundle (default {BUNDLED_MODEL})")
    ap.add_argument("--stage-only", action="store_true",
                    help="Only stage Ollama+model (skip the PyInstaller build)")
    args = ap.parse_args()

    if STAGING.exists():
        shutil.rmtree(STAGING)
    print("Staging Ollama + model…")
    stage_ollama()
    stage_model(args.model)
    if args.stage_only:
        print("Staged only (skipped build).")
        return 0
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
