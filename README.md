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

1. Install [Ollama](https://ollama.com) and pull the model:
   ```
   ollama pull qwen3:8b
   ```
2. Get PitEngineer:
   - **Packaged:** run `dist/PitEngineer.exe` (build it with
     `pip install pyinstaller && python build_exe.py`), or
   - **From source:** `pip install -r requirements.txt` then
     `python PitEngineer.py`
3. Start Assetto Corsa, get on track, and in PitEngineer press **Detect car**.
4. Press **Start stint**, drive a few laps, press **Stop & analyze**.
5. Read the debrief, review the proposed change, press **Apply change & continue**.
6. **Reload the setup in the pits** (re-enter the garage / re-select the setup)
   so AC applies it, then drive the next stint. Repeat until it's dialled in.

### Option B — the command line

```
python -m pitengineer.autotune      # auto-detects car, track, and setup
```
Same loop, in a terminal. Drive → press Enter to stop → debrief → `Y` to apply.

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

## Requirements & notes

- **Windows + Assetto Corsa** (the telemetry uses AC's Windows Shared Memory).
- **Ollama** with a capable model (`qwen3:8b` recommended). Diagnosis takes
  ~30–60s on CPU; that's fine between stints.
- AC can't hot-swap a setup mid-lap — you reload it in the pits, so the loop is
  a between-stints debrief, not a live overlay.
- A faster stint can be *you* driving better, not the setup — PitEngineer uses
  your consistency to rate confidence and avoid chasing noise.

Built from a disaster race and the wish that something had told me *why*.
