# PitEngineer

**An AI race engineer for Assetto Corsa.** Drive a few laps and PitEngineer reads
your live telemetry, works out what the car is doing wrong *and* how **you**
drive, and tunes the setup to suit you — a debrief and a setup change after every
stint, until the car is dialled in.

It runs **locally and free** on your own machine (via [Ollama](https://ollama.com)),
so there's no subscription and no API key required. Optional Claude backend for
maximum quality.

> Think of it as the setup half nobody built: CrewChief spots for you and
> TrackTitan coaches your driving — PitEngineer tunes your **car**.

---

## What makes it different

ChatGPT/Gemini can hand out generic setup advice, but they can't see your game.
PitEngineer reads Assetto Corsa's live telemetry — your tyre temps, slip, inputs
and lap times — so its advice is grounded in what your car actually did:

- **Auto-detects your car and track** from the running game — works with *any*
  car or mod, no configuration. It learns each car's real adjustable parameters
  from your own setups.
- **Diagnoses from data, not vibes** — "moderate understeer, front axle
  overworked (fronts 12°C hotter)", read straight from your driving.
- **Deep, guide-backed analysis** — dynamic camber under load, hot tyre
  **pressures vs the ~26–28 psi window**, lock-ups and wheelspin (brakes/diff),
  bottoming over kerbs, and where on the lap you lose time — each mapped to the
  right lever using established race-engineering principles.
- **Personalised to your style** — smooth vs aggressive, trail-braker vs not.
  The same car gets a different setup for you than for someone else.
- **An iterative auto-tune loop** — it proposes a change, you apply it, drive
  again, and it judges whether it helped (lap time **and** balance, with a
  confidence rating) before the next step. When gains plateau, it says so.
- **Honest** — if the car is balanced and you're still slow, it tells you it's
  your driving, not the setup.
- **Safe** — every proposed value is validated against the car's real ranges,
  and your original setup is backed up before any change.

---

## Quick start

### Option A — the app (recommended)

1. Download and unzip the latest release, or build it:
   ```
   python build_standalone.py
   ```
   This produces a self-contained folder with Ollama bundled in.

2. Pull the AI model (Ollama will download ~2GB):
   ```
   ollama pull qwen3:8b
   ```
   Or use the bundled Ollama that starts automatically when you launch PitEngineer.

3. *(Optional)* If your Assetto Corsa setups live outside the default Documents
   folder (e.g., OneDrive):
   - Click **Setups folder** (top-right button next to Detect car)
   - Browse to your custom folder and confirm
   - The app will show the selected path and reset for re-detection

4. Start Assetto Corsa, get on track, and in PitEngineer press **Detect car**.
5. Press **Start stint**, drive a few laps, press **Stop & analyze**.
6. Read the debrief, review the proposed change, press **Apply change & continue**.
7. **Reload the setup in the pits** (re-enter the garage / re-select the setup)
   so AC applies it, then drive the next stint. Repeat until it's dialled in.

### Option B — the command line

```
python -m pitengineer.autotune      # auto-detects car, track, and setup
```

This requires:
- **Ollama installed locally** ([download](https://ollama.com))
- Model pulled: `ollama pull qwen3:8b`
- Dependencies: `pip install -r requirements.txt`

Same loop, in a terminal. Drive → press Enter to stop → debrief → `Y` to apply.

---

## Using a custom setups directory

If your Assetto Corsa setups live outside the default `Documents/Assetto Corsa/setups`
folder (e.g., redirected to OneDrive, network drive, or custom location), use one
of these methods:

### Method 1 — GUI button (easiest)

1. Launch PitEngineer
2. Click **Setups folder** button (top-right, next to Detect car)
3. Browse to your custom Assetto Corsa setups folder and confirm
4. The path will show in the top bar
5. Click **Detect car** to scan your setups from the new location

### Method 2 — Command line argument

```
python -m pitengineer.autotune --setups-dir "D:\OneDrive\Documents\Assetto Corsa\setups"
```

Or with the GUI:
```
python -m pitengineer.gui --setups-dir "D:\OneDrive\Documents\Assetto Corsa\setups"
```

### Method 3 — Environment variable (persistent)

Set once, use everywhere. On Windows:

**PowerShell:**
```powershell
$env:PITENGINEER_SETUPS_DIR = "D:\OneDrive\Documents\Assetto Corsa\setups"
python -m pitengineer.autotune
```

**Command Prompt:**
```cmd
set PITENGINEER_SETUPS_DIR=D:\OneDrive\Documents\Assetto Corsa\setups
python -m pitengineer.autotune
```

**Persistent (all future sessions):**
```powershell
[Environment]::SetEnvironmentVariable("PITENGINEER_SETUPS_DIR", "D:\OneDrive\Documents\Assetto Corsa\setups", "User")
```
Then restart your terminal and PitEngineer will always use that folder.

---

## Step-by-step workflow

1. **Open Assetto Corsa** and start a session (get on track)
2. **Open PitEngineer** (the app)
3. **Click "Detect car"** (top-right button) — it will scan your setups and learn the car's parameters
4. If your setups are in a custom folder (not default Documents):
   - Click **"Setups folder"** button first
   - Browse and select your custom folder
   - Then click **"Detect car"** again
5. **Click "● Start stint"** (red button, bottom-left)
6. **Drive several laps** in Assetto Corsa (the app is recording your telemetry)
7. **Click "■ Stop & analyze"** when you finish
8. **Wait 30–60 seconds** for the AI to process your data
9. **Read the DETAILS log** (debrief of what the car did)
10. **Review the proposed change** in the PROPOSED CHANGE card (upper-center)
11. **Click "Apply change & continue"** (red button) if you agree
12. **Go back to Assetto Corsa** and reload the setup in the pits:
    - Enter the garage / pause
    - Go to Setup menu
    - Load the setup it created (usually `pitengineer.ini`)
13. **Exit the pits and drive another stint**
14. **Repeat steps 5–13** until the car feels dialled in (or gains plateau)

---

### Best quality (optional): Claude instead of local

```
# put ANTHROPIC_API_KEY in a .env file, then:
python -m pitengineer.autotune --engine claude
```
Pay-as-you-go, a few cents per diagnosis — no subscription. Ollama stays the
free default.

---

## Also included

- **Live telemetry monitor** — verify the sensor works:
  `python -m pitengineer.monitor` (with AC on track).
- **Text-complaint mode** — describe a problem instead of driving:
  `python -m pitengineer.app --setup <file> --manifest <car.json>`.
- **Offline test** — no AC/AI needed: `python -m tests.test_offline`.

---

## How it works

```
you drive a stint ─► telemetry reader (Shared Memory API)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   balance /         driver-style       lap / sector
   tyre temps        profile            times
        └─────────────────┼─────────────────┘
                          ▼
             stint report + "did the last change help?"
                          ▼
        AI engineer (Ollama qwen3:8b, or Claude)  ◄─ car's real params
                          ▼                            + dynamics grounding
             validated, in-range setup change(s)
                          ▼
        you Apply ─► setup .ini written (original backed up)
                          ▼
             reload in pits ─► drive again ─► it refines
```

The AI **advises**; PitEngineer's code **guarantees legality** — every value is
clamped to the car's real min/max/step and hallucinated parameters are dropped.

| Module | Role |
| --- | --- |
| `shared_memory.py` | Read live AC telemetry (physics/graphics/static) |
| `car_data.py` | Learn any car's parameters + ranges from your setups |
| `summarizer.py` | Telemetry → understeer/oversteer + tyre balance |
| `driver_profile.py` | Driving style from your inputs |
| `stint.py` | Record + analyse a stint (laps, balance, style) |
| `session_log.py` | Per car/track memory + "did the change help?" verdict |
| `translator.py` | The AI diagnosis, validated against the car |
| `engines.py` | Pluggable backend: Ollama (default) / Claude |
| `autotune.py` | The auto-tune stint loop (CLI) |
| `gui.py` | The desktop app window |

---

## Changelog

### 1.1.0 — Configurable setups directory

- **New:** GUI "Setups folder" button to pick a custom Assetto Corsa setups root
  at runtime (solves OneDrive/redirected Documents issues).
- **New:** `--setups-dir` CLI option and `PITENGINEER_SETUPS_DIR` environment
  variable for headless or scripted usage.
- Setups discovery, car detection, and save targets now respect the chosen folder.
- Desktop builds auto-refresh during rebuild, so the GUI stays current.
- Gearing issues stay at the front of auto-tune proposals when telemetry shows the car is gearing-limited.
- The selected setups folder persists between app launches.
- Desktop release updates refresh more reliably even with bundled Ollama files in use.

### 1.0.0 — Initial release

- Auto-tune loop with live telemetry analysis (telemetry → diagnosis → setup change).
- AI-driven setup optimization (Ollama or Claude backend).
- Iterative one-change-per-stint or full-setup-pass modes.
- Safe, validated changes (clamped to car's legal ranges).
- Desktop app + CLI + standalone offline bundle.

---

## Requirements & notes

- **Windows + Assetto Corsa** (the telemetry uses AC's Windows Shared Memory).
- **Ollama** — bundled in the packaged release (auto-starts on launch). If running
  from source, install [Ollama](https://ollama.com) separately and pull a model
  (`ollama pull qwen3:8b`, ~2GB).
- Diagnosis takes ~30–60s on CPU; that's fine between stints.
- AC can't hot-swap a setup mid-lap — you reload it in the pits, so the loop is
  a between-stints debrief, not a live overlay.
- A faster stint can be *you* driving better, not the setup — PitEngineer uses
  your consistency to rate confidence and avoid chasing noise.

Built from a disaster race and the wish that something had told me *why*.
