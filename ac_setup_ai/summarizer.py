"""Turn raw driving telemetry into a compact, data-derived diagnosis.

This is the wire between the sensor and the brain. Instead of the driver typing
"it understeers", we capture a window of live telemetry while they drive and
compute the tell-tale signals ourselves:

* balance tendency (understeer / oversteer) from front-vs-rear tyre slip
* tyre temperature balance (front/rear and left/right)
* how hard the car is being driven (slip magnitude, brake/throttle usage)

The result is a plain-language summary the AI diagnoses from — no typing.

Wheel/tyre arrays are ordered [FL, FR, RL, RR] throughout (AC's order).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .shared_memory import ACTelemetry, PhysicsSnapshot


@dataclass
class TelemetrySummary:
    samples: int
    driving_samples: int
    # Mean tyre core temps per corner [FL, FR, RL, RR]
    tyre_temp: tuple[float, float, float, float]
    front_temp: float
    rear_temp: float
    left_temp: float
    right_temp: float
    # Mean slip per axle (higher = more sliding)
    front_slip: float
    rear_slip: float
    tendency: str            # "understeer" | "oversteer" | "neutral"
    tendency_strength: str   # "mild" | "moderate" | "strong"
    avg_speed_kmh: float
    max_speed_kmh: float
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Plain-language, number-backed summary for the AI to reason over."""
        fl, fr, rl, rr = self.tyre_temp
        lines = [
            f"Captured {self.driving_samples} driving samples.",
            f"Handling balance: {self.tendency_strength} {self.tendency} "
            f"(front slip {self.front_slip:.3f} vs rear slip {self.rear_slip:.3f}).",
            f"Tyre core temps (C): FL {fl:.0f}, FR {fr:.0f}, RL {rl:.0f}, RR {rr:.0f}.",
            f"Front avg {self.front_temp:.0f}C vs rear avg {self.rear_temp:.0f}C "
            f"(front is {self.front_temp - self.rear_temp:+.0f}C vs rear).",
            f"Left avg {self.left_temp:.0f}C vs right avg {self.right_temp:.0f}C "
            f"(left is {self.left_temp - self.right_temp:+.0f}C vs right).",
            f"Speed: avg {self.avg_speed_kmh:.0f} km/h, max {self.max_speed_kmh:.0f} km/h.",
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(samples: list[PhysicsSnapshot]) -> TelemetrySummary:
    """Compute the summary from captured physics snapshots."""
    # Keep only genuine driving samples: moving with real load on the tyres.
    driving = [s for s in samples if s.speed_kmh > 20.0]
    src = driving or samples  # fall back so we never divide by zero

    fl_t = _mean([s.tyre_core_temp[0] for s in src])
    fr_t = _mean([s.tyre_core_temp[1] for s in src])
    rl_t = _mean([s.tyre_core_temp[2] for s in src])
    rr_t = _mean([s.tyre_core_temp[3] for s in src])

    front_temp = (fl_t + fr_t) / 2
    rear_temp = (rl_t + rr_t) / 2
    left_temp = (fl_t + rl_t) / 2
    right_temp = (fr_t + rr_t) / 2

    # Slip per axle over driving samples. AC wheelSlip grows with sliding, so a
    # higher-slipping axle is the one giving up grip first.
    front_slip = _mean([(s.wheel_slip[0] + s.wheel_slip[1]) / 2 for s in src])
    rear_slip = _mean([(s.wheel_slip[2] + s.wheel_slip[3]) / 2 for s in src])

    tendency, strength = _classify_balance(front_slip, rear_slip)

    notes: list[str] = []
    if front_temp - rear_temp > 12:
        notes.append("Front tyres running much hotter than rear - front axle is overworked.")
    elif rear_temp - front_temp > 12:
        notes.append("Rear tyres running much hotter than front - rear axle is overworked.")

    return TelemetrySummary(
        samples=len(samples),
        driving_samples=len(driving),
        tyre_temp=(fl_t, fr_t, rl_t, rr_t),
        front_temp=front_temp,
        rear_temp=rear_temp,
        left_temp=left_temp,
        right_temp=right_temp,
        front_slip=front_slip,
        rear_slip=rear_slip,
        tendency=tendency,
        tendency_strength=strength,
        avg_speed_kmh=_mean([s.speed_kmh for s in src]),
        max_speed_kmh=max((s.speed_kmh for s in src), default=0.0),
        notes=notes,
    )


def _classify_balance(front_slip: float, rear_slip: float) -> tuple[str, str]:
    """Front vs rear slip -> understeer/oversteer + how pronounced."""
    if front_slip <= 0 and rear_slip <= 0:
        return "neutral", "mild"
    denom = max(front_slip, rear_slip, 1e-6)
    ratio = (front_slip - rear_slip) / denom  # >0 front slides more (understeer)

    if ratio > 0.08:
        tendency = "understeer"
    elif ratio < -0.08:
        tendency = "oversteer"
    else:
        return "neutral", "mild"

    mag = abs(ratio)
    strength = "strong" if mag > 0.30 else "moderate" if mag > 0.15 else "mild"
    return tendency, strength


def capture(duration_s: float = 30.0, rate_hz: float = 20.0,
            tele: ACTelemetry | None = None) -> list[PhysicsSnapshot]:
    """Sample live physics for `duration_s` seconds. Requires AC on track.

    Opens its own telemetry connection unless one is passed in.
    """
    own = tele is None
    tele = tele or ACTelemetry().open()
    period = 1.0 / rate_hz
    samples: list[PhysicsSnapshot] = []
    end = time.monotonic() + duration_s
    try:
        while time.monotonic() < end:
            samples.append(tele.read_physics())
            time.sleep(period)
    finally:
        if own:
            tele.close()
    return samples
