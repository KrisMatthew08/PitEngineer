# Assetto Corsa Setup AI — Translator (Phase 1 MVP)

Describe how your car feels wrong in plain language; get back **validated,
in-range** setup changes with a *why* for each, and apply them to your setup
`.ini` with a single confirmation. Built for the original **Assetto Corsa**.

See [DESIGN.md](DESIGN.md) for the full concept and roadmap. This repo is
Phase 1 — the Translator. Live telemetry and driver personalization come later.

## How it works

```
your complaint ──► AI translator (Claude, tool-use) ──► proposed changes
                          │                                    │
              symptom→lever knowledge base            validated + clamped
              + current setup + legal ranges          against the car manifest
                                                              │
                                                    before/after diff ──► Apply (writes .ini, backs up original)
```

The AI **advises**; our code **guarantees legality** — every proposed value is
clamped to the car's real min/max/step before you ever see it, and hallucinated
parameters are dropped. Your original setup is backed up before any write.

## Setup

1. Install the dependency:
   ```
   pip install -r requirements.txt
   ```
2. Provide an Anthropic API key (either works):
   - copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`, **or**
   - run `ant auth login` and leave the env var unset.

## Run it (against the bundled sample)

```
python -m ac_setup_ai.app --setup data/sample_setup.ini --manifest data/manifests/generic_gt3.json
```

Then type things like:
- `the rear steps out when I get on the power out of slow corners`
- `car won't turn in on corner entry`
- `it's unstable and locks the rears under braking`

You'll get a diagnosis, a ranked before/after diff, and a `[y/N]` apply prompt.

## Run it against your real car

1. Find a setup: `Documents/Assetto Corsa/setups/<car_id>/<track_id>/<name>.ini`
2. Write a manifest for that car (copy `data/manifests/generic_gt3.json` and fill
   in the real adjustable parameters and their min/max/step). The parameter
   section names must match the ones in the `.ini`.
3. Point the app at both:
   ```
   python -m ac_setup_ai.app --setup "<path to your .ini>" --manifest "<your manifest>.json"
   ```
4. After applying, **re-select / reload the setup in the pits** for it to take effect.

## Project layout

| File | Role |
| --- | --- |
| `ac_setup_ai/setup_file.py` | Parse + safely write AC setup `.ini` (index space, backups) |
| `ac_setup_ai/manifest.py` | Car adjustability manifest — the legal-range guardrail |
| `ac_setup_ai/knowledge.py` | Symptom → cause → lever grounding for the model |
| `ac_setup_ai/translator.py` | The AI layer: complaint → validated changes (tool-use) |
| `ac_setup_ai/app.py` | CLI: chat, diff, apply |
| `data/` | Sample setup + a generic GT3 manifest |

## Status & limitations

- **Index space:** changes are made in AC's index space. Human-readable values
  (psi/deg/mm) shown for a parameter are only as accurate as the manifest's
  `value_at_min` / `value_per_step` mapping — approximate on the sample GT3.
- **One car manifest so far** (a generic GT3). Real per-car manifests are the
  main thing to add for accuracy.
- **No telemetry yet** — this is the pure Translator. Phase 2 adds the Shared
  Memory reader so recommendations can use real tyre temps and slip data.
