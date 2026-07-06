# Assetto Corsa Setup AI — Design Doc

> An AI setup coach for Assetto Corsa. The driver describes what the car feels
> wrong in plain language; the app translates that into concrete, valid setup
> changes tailored to **that driver**, applies them to the game, and learns from
> the telemetry that comes back.

Status: **Design** · Owner: you · Last updated: 2026-07-06

---

## 1. Product vision

Tuning a car in Assetto Corsa means facing ~40 interacting parameters (tyre
pressures, ARBs, springs, dampers, camber, toe, diff, aero, gearing, brake
bias...). Most players know the car *feels* wrong but can't map that feeling to
the right knob. The expert loop is:

> drive → "loose on exit" → know it's the rear ARB / diff / rear wing → change →
> re-test

This app collapses that loop for non-experts and speeds it up for everyone.

**North-star:** the driver types *"the rear steps out when I get on the power
out of slow corners"* and gets back a ranked list of changes with values,
an explanation of *why*, and a one-click **Apply** — personalized to how they
actually drive.

### The three products (we build them as layers, not forks)

| Layer | What it does | Phase |
| --- | --- | --- |
| **Translator** (front door) | Natural-language symptom → setup changes + *why* | 1 (MVP) |
| **Data Optimizer** | Reads live telemetry, diagnoses from real data | 2 |
| **Autonomous / personalized coach** | Learns driver style, converges on optimal setup | 3 |

The Translator is the UX everyone sees. Phases 2–3 make its recommendations
smarter by feeding it telemetry and a driver profile — same interface, better brain.

---

## 2. Assetto Corsa integration (the technical crux)

We chose **read + write everything**. Three integration surfaces:

### 2.1 Live telemetry — Shared Memory API (read)
AC publishes real-time telemetry via Windows **memory-mapped files**:
- `acpmf_physics` — tyre temps (core/inner/middle/outer), tyre pressures, slip
  ratio/angle per wheel, suspension travel, G-forces, wheel speeds, inputs
  (throttle/brake/steer/clutch), ride height.
- `acpmf_graphics` — session state, current/last/best lap, sector, position,
  tyre compound, fuel.
- `acpmf_static` — car model, track, max RPM, tyre count, per-car limits.

Read at ~20–60 Hz. This is the goldmine for Phase 2 diagnosis (e.g. "front
tyres 15°C hotter than rear + high front slip angle → understeer, and it's a
*setup* problem not a *driving* problem").

### 2.2 Setup files (read + write) — the closed loop
Setups live as `.ini` at:
```
Documents/Assetto Corsa/setups/<car_id>/<track_id>/<name>.ini
```
Each key is an **index** into an allowed range, not a raw value. Example:
```
[PRESSURE_LF]
VALUE=8      ; index; real psi = f(index) per car
[ARB_FRONT]
VALUE=3
```
Because we can **write** these files, the app closes the loop:
recommend → write .ini → driver reloads setup in-game → telemetry returns → refine.

### 2.3 Car constraints — what's actually adjustable
Every car exposes which params are adjustable and their **min/max/step** (from
the car's `data.acd` / `setup.ini`). The AI must only ever propose valid,
in-range index values. We cache a per-car "adjustability manifest" so the model
can't hallucinate an impossible setting.

> ⚠️ Writing files while AC has the setup loaded: the driver must re-select /
> reload the setup in the pits for changes to take. We surface that clearly.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  UI  (chat-style Translator + setup diff + Apply button)      │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│  Core / Orchestrator                                          │
│   • builds the AI request (symptom + setup + constraints +    │
│     telemetry summary + driver profile)                       │
│   • validates AI output against the car's adjustability manifest│
│   • produces a setup diff, applies on confirm                 │
└──────┬───────────────┬───────────────┬──────────────┬─────────┘
       │               │               │              │
┌──────▼─────┐  ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
│ AC Telemetry│  │ Setup File  │ │ Car Manifest│ │  AI Layer   │
│  Reader     │  │ Read/Write  │ │  (ranges)   │ │ (Claude,    │
│ (shared mem)│  │  (.ini)     │ │             │ │  tool-use)  │
└─────────────┘  └─────────────┘ └─────────────┘ └─────────────┘
                                                        │
                                                 ┌──────▼──────┐
                                                 │ Driver      │
                                                 │ Profile /   │
                                                 │ Setup history│
                                                 └─────────────┘
```

### 3.1 The AI Layer (what makes it "AI")
- Uses Claude with **structured tool-use output** so it returns a *machine-valid*
  list of changes, not prose we have to parse. Schema per change:
  `{ param, current_index, proposed_index, direction, magnitude, reason,
     confidence }`.
- Input to the model each turn:
  1. Driver's natural-language complaint.
  2. Current setup (decoded to human values + raw indices).
  3. The car's adjustability manifest (valid ranges/steps).
  4. *(Phase 2)* A compact telemetry summary (tyre temp deltas, slip, where in
     the lap the issue occurs).
  5. *(Phase 3)* The driver profile (smooth vs aggressive, brake/throttle style).
- The **orchestrator validates** every proposed index against the manifest and
  clamps/ rejects out-of-range values before showing them. The model advises;
  our code guarantees legality.
- Model choice: default to **Claude Opus 4.8** (`claude-opus-4-8`) for the
  reasoning-heavy diagnosis; **Claude Sonnet 5** (`claude-sonnet-5`) as a
  cheaper/faster option for simple Translator queries. (Confirm current pricing/
  ids via the claude-api skill at build time.)

### 3.2 The domain knowledge (the "translator brain")
A curated symptom → cause → parameter map seeds the model and keeps it honest.
Sketch:

| Symptom (driver words) | Likely cause | First levers |
| --- | --- | --- |
| Loose / oversteer on power out | Too much rear grip loss / diff | Soften rear ARB, reduce power-side diff, add rear wing, rear pressure |
| Won't turn in / understeer entry | Front grip / balance | Soften front ARB, more front camber, front toe-out, brake bias fwd |
| Snappy mid-corner | Aero/mechanical balance | Springs, camber, ride height, ARB balance |
| Tyres overheating (one end) | Pressure/camber/load | Adjust pressure, camber, ARB |
| Unstable under braking | Bias / rear stability | Brake bias, rear ride height, diff coast, rear ARB |

This lives as structured data the model can reference — it's the difference
between a lookup table and an AI that reasons over the driver's specific car,
telemetry, and history.

### 3.3 Driver personalization (the "for the driver" part)
Phase 3. From telemetry inputs we derive a profile:
- **Smoothness** (steering rate, throttle/brake modulation)
- **Aggression** (peak brake, corner-entry speed, trail-braking)
- **Where they lose time** (sector/segment vs a reference)

Two drivers, same car+track → different recommendations. Smooth driver tolerates
a pointier car; aggressive driver needs stability. The profile is an extra input
to the AI layer and a bias on which levers we prefer.

---

## 4. Tech stack (recommendation)

| Concern | Choice | Why |
| --- | --- | --- |
| Language / runtime | **Python** | Trivial Shared Memory access (`mmap` + `ctypes` struct), first-class Anthropic SDK, easy .ini handling |
| Shared Memory | `mmap` + `ctypes` structs mirroring the AC layout | Standard approach; well-documented struct layouts exist |
| UI | Start with a local web UI (FastAPI + simple frontend) *or* a lightweight desktop shell | Chat UX is natural; can wrap in a window later |
| AI | Anthropic Python SDK, tool-use for structured output | Guarantees valid, parseable change lists |
| Storage | Local JSON/SQLite for setup history + driver profile | Offline-first, no account needed |

Alternative if you'd rather ship a polished desktop app immediately: **Electron
+ TypeScript** (Anthropic TS SDK; Shared Memory via a small native/node addon).
Python is faster to a working prototype; Electron is nicer to distribute.

---

## 5. MVP scope (Phase 1 — The Translator)

**Goal:** driver picks car+track, types a complaint, gets valid ranked changes
with explanations, and can Apply them to the setup file.

Must-have:
1. Load an existing setup `.ini` for a chosen car/track.
2. Load that car's adjustability manifest (valid ranges).
3. Chat box: driver describes the problem.
4. AI returns validated, in-range changes + per-change reasoning.
5. Show a **before/after diff**; **Apply** writes the `.ini`.
6. Clear "reload the setup in-game" instruction after applying.

Explicitly **out** of MVP (comes in Phase 2+): live telemetry reading,
auto-diagnosis, driver profiling, autonomous iteration.

### Success test
On a known car (e.g. a popular GT3), for 5 classic complaints (power-on
oversteer, entry understeer, braking instability, tyre overheating,
too-nervous), the app proposes changes a competent sim racer would agree with,
all in-range, with sensible explanations.

---

## 6. Open questions / risks

- **Struct layout drift:** the Shared Memory struct must match the AC/ACC build.
  Mitigation: keep the struct definition in one versioned module; validate on
  startup by sanity-checking known fields.
- **Index → real value decoding:** per-car mapping from index to psi/°/mm. We
  need a reliable source (car data or a calibration table). MVP can work in pure
  index space and still be correct; human-readable values are a nice-to-have.
- **AC vs ACC:** ~~Which one?~~ **Resolved: original Assetto Corsa.** Setup
  `.ini` index format and the `acpmf_*` struct layouts in this doc apply. ACC
  (JSON setups, different structs) is explicitly out of scope.
- **AI trust:** always show reasoning + keep the driver in the loop with a diff
  and an explicit Apply. Never silently overwrite a setup; back up the original.

---

## 7. Roadmap

- **Phase 1 — Translator MVP:** load setup + manifest, chat, validated changes,
  diff, apply. *(This is what we build first.)*
- **Phase 2 — Data Optimizer:** Shared Memory reader, telemetry summarizer,
  auto-diagnosis feeding the Translator.
- **Phase 3 — Personalized coach:** driver profile from inputs, personalized
  recommendations, setup history & A/B comparison.
- **Phase 4 — Autonomous tuner:** guided iterative loop that converges on an
  optimal setup for the driver.
```
