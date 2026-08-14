# PitEngineer

PitEngineer is a local AI race engineer for **Assetto Corsa**. It reads your
live driving telemetry, explains the car's handling balance, and suggests safe
setup changes for the next stint.

This folder is self-contained: **do not move `PitEngineer.exe` out of this
folder**. The `_internal` folder contains the files the app needs to run.

## Before you start

1. Start Assetto Corsa and enter a session with the car and track you want to
   tune.
2. Keep Assetto Corsa running in the background.
3. Double-click `PitEngineer.exe`.

No Python, Ollama installation, internet connection, subscription, or API key
is required for this version.

## Typical tuning session

1. In PitEngineer, choose **Detect car** while you are on track.
2. Choose **Start stint** and drive several representative laps. Avoid using a
   single out-lap, traffic lap, or mistake as the only data point.
3. Choose **Stop & analyze**. The app reviews lap times, tyre behaviour,
   balance, driving inputs, and sectors.
4. Read the debrief and proposed setup change. You stay in control: nothing is
   changed until you approve it.
5. Choose **Apply change & continue** if you want to use the recommendation.
   Your original setup is backed up automatically.
6. Back in Assetto Corsa's pits/garage, reload or re-select the setup so AC
   applies the edited file.
7. Drive another stint. PitEngineer compares the result and refines the setup
   only when the evidence supports it.

## Important notes

- Assetto Corsa cannot apply setup edits while you drive. Always reload the
  setup in the pits before judging the next stint.
- Setup advice is restricted to parameters and ranges that the detected car
  actually supports; invalid or invented settings are ignored.
- A quicker lap is not always a setup gain. Consistent stints give the app the
  clearest answer.
- The first AI analysis can take a little longer while the bundled local model
  starts. Later analyses should be faster.

## If something does not work

**PitEngineer cannot detect the car or telemetry**

- Confirm Assetto Corsa is running and you are in an active on-track session.
- Start the app after entering the session, then try **Detect car** again.

**A setup change does not appear in-game**

- Return to the pits or garage and re-select/reload the setup. AC does not
  hot-swap setup files mid-lap.

**Windows shows a SmartScreen or antivirus warning**

- This can happen with newly built, unsigned Windows applications. Verify that
  you received the folder from a source you trust before choosing to run it.

**The app will not start after copying it**

- Restore the complete folder, including `_internal`. Do not copy only the
  `.exe` file.

## Sharing this app

To share PitEngineer, zip the entire folder. Recipients should extract the zip
before opening `PitEngineer.exe`; running it directly from a compressed zip can
prevent the bundled components from working correctly.
