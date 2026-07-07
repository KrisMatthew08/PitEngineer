# PitEngineer — self-contained offline build

This branch (`offline-standalone`) builds a version of PitEngineer that needs
**nothing installed** — no Python, no Ollama, no internet, no API key. The AI is
bundled in: `ollama.exe` + a small model ride inside the package, and the app
auto-starts them on launch.

## How it works
- `pitengineer/ollama_manager.py` finds the bundled `ollama.exe` (or a system
  install as fallback), starts `ollama serve` in the background pointed at the
  bundled model store (`OLLAMA_MODELS`), and waits until it's reachable.
- The GUI calls this on startup; the existing `OllamaEngine` then talks to
  `localhost:11434` exactly as before.
- The app auto-selects **whatever model is bundled** (`bundled_model_name()`),
  so there's no mismatch.
- Analysis, corner time-loss, and **Full Setup Pass** are rule-based and work
  even if the AI never starts — only Quick Tune needs the model.

## Building it
Requires Ollama installed with the model pulled (the builder copies only that
model's data, not your whole store), plus PyInstaller.

```
pip install pyinstaller
ollama pull llama3.2:3b            # or gemma3:1b for a smaller bundle
python build_standalone.py --model llama3.2:3b
```

Output: `dist/PitEngineer/` — a **folder** (onedir), not a single file, because
the model is ~1–2 GB and a onefile exe would re-extract it to temp every launch.

- Bundle size: ~119 MB Ollama runtime (CPU + Vulkan; the huge CUDA/ROCm GPU
  folders are deliberately skipped) **+** the model
  (gemma3:1b ≈ 0.8 GB, llama3.2:3b ≈ 2 GB).

## Distributing it
Zip `dist/PitEngineer/` and share it (or attach to a GitHub Release). The user
unzips and runs `PitEngineer.exe` inside — fully offline, nothing to install.

## Model choice
- `llama3.2:3b` — better Quick Tune reasoning, ~2 GB bundle. Recommended default.
- `gemma3:1b` — smallest (~0.8 GB), weaker Quick Tune reasoning, but Full Setup
  Pass (rule-based) is unaffected.

## Validating the bundle
Because your dev machine already runs Ollama on port 11434, the bundled copy
won't be exercised there (the app finds the running server first). To truly test
the bundled Ollama, run `dist/PitEngineer/PitEngineer.exe` on a machine (or user)
**without** Ollama installed - it should start the bundled server and load the
bundled model on its own.
