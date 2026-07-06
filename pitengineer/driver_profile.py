"""Infer a driver's style from their telemetry inputs.

The same car+track wants a different setup for a smooth, consistent driver than
for an aggressive, erratic one. We read style from the inputs themselves:

* smoothness  - how progressively the driver works the wheel/pedals
* aggression  - how hard/fast they attack the brakes and throttle
* trail_brake - how much they brake while turning
* consistency - lap-to-lap repeatability (from lap times)

gas/brake are 0..1 in AC (reliable). Steering units vary by car, so steering is
measured *relative* to the driver's own max in the stint - unit-agnostic.

Each trait is a 0..1 score plus a plain-language label the AI can use to bias
its setup recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from .shared_memory import PhysicsSnapshot


@dataclass
class DriverProfile:
    smoothness: float      # 0 erratic .. 1 very smooth
    aggression: float      # 0 gentle .. 1 very aggressive
    trail_brake: float     # 0 none .. 1 heavy trail-braking
    consistency: float     # 0 erratic laps .. 1 metronomic
    labels: dict[str, str] # trait -> word

    def describe(self) -> str:
        return (
            f"Driving style: {self.labels['smoothness']} inputs, "
            f"{self.labels['aggression']} on the brakes/throttle, "
            f"{self.labels['trail_brake']} trail-braking, "
            f"{self.labels['consistency']} lap-to-lap.\n"
            f"(smoothness {self.smoothness:.2f}, aggression {self.aggression:.2f}, "
            f"trail-brake {self.trail_brake:.2f}, consistency {self.consistency:.2f})"
        )

    def setup_bias(self) -> str:
        """A short instruction to the AI on how to bias the setup for this driver."""
        bits: list[str] = []
        if self.smoothness > 0.6 and self.consistency > 0.5:
            bits.append(
                "Driver is smooth and consistent - they can handle a pointier, "
                "more responsive car; don't over-prioritise stability."
            )
        if self.aggression > 0.6 or self.consistency < 0.4:
            bits.append(
                "Driver is aggressive or inconsistent - favour a stable, "
                "forgiving rear so the car doesn't punish mistakes."
            )
        if self.trail_brake > 0.5:
            bits.append(
                "Driver trail-brakes heavily - prioritise rear stability under "
                "braking (brake bias, diff coast, rear support)."
            )
        if not bits:
            bits.append(
                "Balanced, moderate driving style - aim for a neutral, "
                "predictable car."
            )
        return " ".join(bits)


def _label(value: float, low: str, mid: str, high: str) -> str:
    return low if value < 0.34 else mid if value < 0.67 else high


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def compute_profile(
    samples: list[PhysicsSnapshot],
    lap_times_ms: list[int] | None = None,
) -> DriverProfile:
    """Derive a DriverProfile from stint samples (and lap times if available)."""
    driving = [s for s in samples if s.speed_kmh > 20.0] or samples

    # --- Smoothness: low sample-to-sample change in steer/throttle/brake ---
    steer_rate = _series_rate([s.steer for s in driving])
    gas_rate = _series_rate([s.gas for s in driving])
    brake_rate = _series_rate([s.brake for s in driving])
    # Normalise steer rate by observed steer range (unit-agnostic).
    steer_vals = [s.steer for s in driving]
    steer_span = (max(steer_vals) - min(steer_vals)) if steer_vals else 1.0
    steer_rate_n = steer_rate / steer_span if steer_span > 1e-6 else 0.0
    # Higher combined rate -> less smooth. Empirical scaling into 0..1.
    roughness = _norm(steer_rate_n + gas_rate + brake_rate, 0.02, 0.30)
    smoothness = 1.0 - roughness

    # --- Aggression: high peak brake + fast brake application ---
    peak_brake = max((s.brake for s in driving), default=0.0)
    hard_brake_frac = _frac(driving, lambda s: s.brake > 0.85)
    fast_throttle = _frac(driving, lambda s: s.gas > 0.95)
    aggression = _norm(
        0.4 * peak_brake + 0.4 * hard_brake_frac + 0.2 * fast_throttle, 0.15, 0.75
    )

    # --- Trail-braking: braking while cornering ---
    corner_thresh = 0.30 * steer_span + (min(steer_vals) if steer_vals else 0.0)
    cornering = [s for s in driving if abs(s.steer) > abs(corner_thresh) * 0.3]
    trail = _frac(cornering, lambda s: s.brake > 0.15) if cornering else 0.0
    trail_brake = _norm(trail, 0.05, 0.45)

    # --- Consistency: from lap-time spread ---
    consistency = _consistency_from_laps(lap_times_ms)

    labels = {
        "smoothness": _label(smoothness, "erratic", "moderately smooth", "very smooth"),
        "aggression": _label(aggression, "gentle", "moderately aggressive", "very aggressive"),
        "trail_brake": _label(trail_brake, "minimal", "some", "heavy"),
        "consistency": _label(consistency, "inconsistent", "fairly consistent", "very consistent"),
    }
    return DriverProfile(smoothness, aggression, trail_brake, consistency, labels)


def _series_rate(values: list[float]) -> float:
    """Mean absolute change between consecutive samples."""
    if len(values) < 2:
        return 0.0
    return mean(abs(b - a) for a, b in zip(values, values[1:]))


def _frac(samples: list, pred) -> float:
    if not samples:
        return 0.0
    return sum(1 for s in samples if pred(s)) / len(samples)


def _consistency_from_laps(lap_times_ms: list[int] | None) -> float:
    # Use ALL laps here (do NOT filter out spins/offs): those ARE the
    # inconsistency. A spin is a real repeatability failure - StintReport then
    # decides whether to blame the car (overheating/imbalance) or the driver.
    laps = [t for t in (lap_times_ms or []) if t and t > 0]
    if len(laps) < 2:
        return 0.5  # unknown -> neutral
    m = mean(laps)
    if m <= 0:
        return 0.5
    cov = pstdev(laps) / m  # coefficient of variation
    # <1% spread -> metronomic (1.0); >4% -> erratic (0.0)
    return 1.0 - _norm(cov, 0.005, 0.04)
