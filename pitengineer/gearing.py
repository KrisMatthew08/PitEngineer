"""Speed, gearing and aero analysis from a stint's telemetry.

Balance (understeer/oversteer) tells you how the car *feels*; this tells you
where the *lap time* is on power tracks and aero cars. From the RPM / gear /
speed we already capture, we work out:

* rev-limiter usage - are you bouncing off the limiter (gears too short)?
* top-gear revs      - do you reach redline in top gear (else gears too tall)?
* top speed          - a proxy for drag / wing level on long straights.

These feed the engineer so it can recommend GEARS and WINGS, not just the ARB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .shared_memory import PhysicsSnapshot

# AC gear convention: 0 = reverse, 1 = neutral, 2 = 1st, ... so on-track drive
# gears are gear >= 2. Display gear number = gear - 1.


@dataclass
class GearingReport:
    max_speed_kmh: float
    max_rpm_seen: int
    car_max_rpm: int
    top_gear: int                 # highest forward gear used (1 = 1st)
    top_gear_peak_rpm: int        # highest rpm reached while in top gear
    rev_limiter_frac: float       # fraction of driving time pinned near redline
    gearing_note: str
    aero_note: str
    issue: str = ""               # "gears_too_short" | "gears_too_tall" | ""
    notes: list[str] = field(default_factory=list)

    def priority_note(self, can_adjust_gears: bool = True) -> str:
        """A hard-priority instruction when there's a clear, costly gearing issue.

        If the car can't change gear ratios, pivot to aero (less wing = less
        drag = higher top speed / reaches redline) instead of chasing a lever
        that doesn't exist.
        """
        if not self.issue:
            return ""
        if not can_adjust_gears:
            return (
                "PRIORITY THIS STINT: the gearing is not ideal, but THIS car "
                "cannot change gear ratios. To gain straight-line speed instead, "
                "reduce WING/drag (if the car has a wing). Do NOT propose gear "
                "changes - they aren't adjustable. Then improve whatever else "
                "helps lap time (tyre pressures, balance)."
            )
        if self.issue == "gears_too_short":
            return (
                "PRIORITY THIS STINT: the car is bouncing off the rev limiter in "
                "top gear - this is losing time on the straight. Fix the GEARING "
                "first (taller final drive / longer top gears) before touching "
                "balance."
            )
        return (
            "PRIORITY THIS STINT: the car never reaches redline in top gear - "
            "the gears are too tall and you're leaving acceleration on the table. "
            "Fix the GEARING first (shorter ratios) before touching balance."
        )

    def describe(self) -> str:
        pct = (self.top_gear_peak_rpm / self.car_max_rpm * 100) if self.car_max_rpm else 0
        lines = [
            f"Top speed: {self.max_speed_kmh:.0f} km/h. Redline ~{self.car_max_rpm} rpm.",
            f"Highest gear used: {self.top_gear}; peak revs in it "
            f"{self.top_gear_peak_rpm} rpm ({pct:.0f}% of redline).",
            f"Time pinned on the rev limiter: {self.rev_limiter_frac*100:.0f}%.",
            f"Gearing: {self.gearing_note}",
            f"Aero: {self.aero_note}",
        ]
        lines.extend(self.notes)
        return "\n".join(lines)


def analyze_gearing(samples: list[PhysicsSnapshot], car_max_rpm: int) -> GearingReport:
    driving = [s for s in samples if s.speed_kmh > 20.0] or samples
    max_rpm = car_max_rpm if car_max_rpm and car_max_rpm > 0 else max(
        (s.rpm for s in driving), default=1)

    max_speed = max((s.speed_kmh for s in driving), default=0.0)
    max_rpm_seen = max((s.rpm for s in driving), default=0)

    # Actual forward gear = raw gear - 1 (AC: 2 -> 1st gear). Ignore N/R.
    forward = [s for s in driving if s.gear >= 2]
    top_gear = max((s.gear - 1 for s in forward), default=0)
    top_gear_samples = [s for s in forward if (s.gear - 1) == top_gear]
    top_gear_peak_rpm = max((s.rpm for s in top_gear_samples), default=0)

    # Bouncing off the limiter: rpm within 2% of redline while on power.
    limiter_thresh = 0.98 * max_rpm
    on_power = [s for s in driving if s.gas > 0.5]
    rev_limiter_frac = (
        sum(1 for s in on_power if s.rpm >= limiter_thresh) / len(on_power)
        if on_power else 0.0
    )

    top_gear_pct = (top_gear_peak_rpm / max_rpm) if max_rpm else 0.0

    # --- Gearing verdict ---
    issue = ""
    if rev_limiter_frac > 0.06 and top_gear_pct > 0.98:
        issue = "gears_too_short"
        gearing_note = (
            "You're spending real time bouncing off the rev limiter in top gear "
            "- the gears are likely too SHORT for this track. A taller final "
            "drive / longer top gears would raise top speed and cut lost time."
        )
    elif top_gear_pct < 0.90:
        issue = "gears_too_tall"
        gearing_note = (
            f"You never reach redline in top gear (peak only {top_gear_pct*100:.0f}%) "
            "- the gears are likely too TALL. Shorter ratios would improve "
            "acceleration and put you at redline by the braking zone."
        )
    else:
        gearing_note = (
            "Gearing looks roughly matched to the track (reaching near redline "
            "in top gear without excessive limiter time)."
        )

    # --- Aero verdict (heuristic; drag vs downforce tradeoff) ---
    if max_speed > 300:
        aero_note = (
            "Very high top speed - on a straight-heavy track this is usually "
            "good, but if you're slow through corners you may be running too "
            "little wing/downforce."
        )
    elif max_speed < 230:
        aero_note = (
            "Modest top speed - if you're getting passed on straights or "
            "hitting the limiter early, you may be carrying too much wing/drag; "
            "trimming wing raises top speed at the cost of some cornering grip."
        )
    else:
        aero_note = (
            "Top speed is in a normal window; wing level is a tradeoff between "
            "straight-line speed and cornering downforce for this track."
        )

    return GearingReport(
        max_speed_kmh=max_speed,
        max_rpm_seen=max_rpm_seen,
        car_max_rpm=max_rpm,
        top_gear=top_gear,
        top_gear_peak_rpm=top_gear_peak_rpm,
        rev_limiter_frac=rev_limiter_frac,
        gearing_note=gearing_note,
        aero_note=aero_note,
        issue=issue,
    )
