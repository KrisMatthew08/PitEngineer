# PitEngineer — Quick Start Guide

Welcome! This is a self-contained, ready-to-use build of PitEngineer. Everything you need is included — no installation required.

---

## Getting Started (5 minutes)

### 1. Extract the ZIP file
Unzip `PitEngineer-standalone-gemma3-1b.zip` to any folder on your computer.

### 2. Run PitEngineer.exe
Double-click `PitEngineer.exe` — it will start immediately. A window should appear.

### 3. Start Assetto Corsa
Launch Assetto Corsa and get on track with the car and track you want to set up.

### 4. Detect Your Car
In the PitEngineer window, click the **"Detect car"** button. It should automatically recognize:
- Your car name
- Your track name
- Your current setup

If detection fails, make sure Assetto Corsa is running and you're on track.

### 5. Drive a Stint
1. Click **"Start stint"** in PitEngineer
2. Drive 3–5 laps in Assetto Corsa (get a feel for the car)
3. Return to PitEngineer and click **"Stop & analyze"**

### 6. Read the Debrief
PitEngineer will analyze your driving and display:
- **Balance issues** (understeer/oversteer)
- **Tyre temperatures** (where the car is working too hard)
- **Your driving style** (smooth, aggressive, trail-braking)
- **A proposed setup change** with confidence rating

### 7. Apply the Change
If you agree with the suggestion:
1. Click **"Apply change & continue"**
2. PitEngineer will save the new setup to your car's setup file
3. **Go back to Assetto Corsa and reload the setup** (exit pits → re-select the setup, or restart session)

### 8. Drive Again & Iterate
1. Drive another stint with the new setup
2. Return to PitEngineer and click **"Stop & analyze"** again
3. Repeat until:
   - Your lap times improve consistently, OR
   - PitEngineer says "setup is balanced, gains have plateaued"

---

## What PitEngineer Does

- **Reads your telemetry** — tire temps, slip angles, your inputs, lap times
- **Diagnoses the problem** — not guesses, actual data-driven analysis
- **Proposes changes** — grounded in real race-engineering principles
- **Validates everything** — every change is checked against the car's real limits
- **Iterates automatically** — measures if each change helped before the next step

---

## Important Notes

### Setup Reloading
Assetto Corsa doesn't hot-swap setups while you're driving. After PitEngineer applies a change:
- **In pits mode:** Exit the garage and re-select the setup
- **Or:** Restart the session
- Then drive again so PitEngineer can measure the impact

### Backup
Your original setup is automatically backed up. PitEngineer will never lose your starting point.

### Confidence Ratings
PitEngineer rates its confidence in each suggestion (High / Medium / Low). Lower confidence means:
- The car might already be balanced
- You might be driving inconsistently
- The change is speculative

### When to Stop
If PitEngineer says "the setup is balanced and you're still slow," it's probably your driving, not the car. That's honest feedback, not a failure.

---

## Troubleshooting

### PitEngineer says "Can't detect car"
- Make sure Assetto Corsa is running
- Get on track (not in menus)
- Try clicking "Detect car" again
- If it still fails, check that AC is fully loaded

### PitEngineer crashes or freezes
- Wait 1–2 minutes (analysis can take time)
- If it's frozen, close it and try again
- The bundled AI model runs on your CPU, so performance depends on your machine

### Setup changes don't apply in AC
- You must **reload the setup in the pits** — AC doesn't update mid-lap
- Exit the garage, re-enter, and re-select the setup
- Or restart the session

### Lap times aren't improving
- Drive consistently — PitEngineer uses your consistency to rate changes
- Give the changes time to work (at least 2–3 stints)
- If PitEngineer says "balanced," focus on your driving instead

---

## Optional: Better AI Quality (Claude)

By default, PitEngineer uses a free local AI model. For even better analysis, you can use Claude (Anthropic's AI):

1. Get an Anthropic API key from [console.anthropic.com](https://console.anthropic.com)
2. Create a file named `.env` in the same folder as `PitEngineer.exe` with:
   ```
   ANTHROPIC_API_KEY=sk-ant-...your-key-here...
   ```
3. Restart PitEngineer — it will now use Claude (pay-as-you-go, a few cents per diagnosis)

---

## More Help

For detailed documentation, visit the GitHub repository:
https://github.com/KrisMatthew08/PitEngineer

---

**Built for people who want to understand why their car is handling poorly — and fix it.**
