"""Live telemetry monitor — the reality check for the Shared Memory reader.

Start Assetto Corsa, get into a session (driving, not the menu), then run:

    python -m ac_setup_ai.monitor

It prints live values ~10x/second. If speed tracks your car, tyre temps sit in
a plausible range (roughly 60-110 C when warm), and throttle/brake move with
your inputs, the struct layout is correct and the sensor works.

Ctrl+C to stop.
"""

from __future__ import annotations

import sys
import time

from .shared_memory import ACTelemetry


def _bar(value: float, width: int = 12) -> str:
    """Render a 0..1 value as a little ASCII bar."""
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def run() -> int:
    try:
        tele = ACTelemetry().open()
    except (OSError, FileNotFoundError):
        print(
            "Could not open AC shared memory.\n"
            "Make sure Assetto Corsa is running and you are IN a session "
            "(on track), not in the main menu.",
            file=sys.stderr,
        )
        return 1

    with tele:
        stat = tele.read_static()
        print(f"Car:   {stat.car_model or '(unknown)'}")
        print(f"Track: {stat.track or '(unknown)'}")
        print(f"Max RPM: {stat.max_rpm}")
        print("\nLive telemetry (Ctrl+C to stop):\n")

        try:
            while True:
                p = tele.read_physics()
                g = tele.read_graphics()

                fl, fr, rl, rr = p.tyre_core_temp
                line = (
                    f"[{g.status:>5}] "
                    f"spd {p.speed_kmh:5.1f} km/h  "
                    f"gear {p.gear - 1:>2}  "  # AC gear: 0=R,1=N,2=1st -> show 1st as 1
                    f"rpm {p.rpm:5d}  "
                    f"thr {_bar(p.gas)} brk {_bar(p.brake)}  "
                    f"tyres FL{fl:5.1f} FR{fr:5.1f} RL{rl:5.1f} RR{rr:5.1f}  "
                    f"lap {g.completed_laps}"
                )
                # \r keeps it on one updating line.
                print(line, end="\r", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
