"""Advanced setup analysis from telemetry: camber, kerbs, brakes/diff, track.

These turn raw channels into concrete, data-backed setup problems the engineer
can act on - the depth that makes PitEngineer more than a generic tool.

* Camber  - the ACTUAL (dynamic) camber the loaded tyre runs while cornering,
            so we can say how much STATIC camber to add/remove.
* Kerbs   - suspension bottoming and wheels going light over kerbs -> springs /
            dampers / bump stops / ride height.
* Brakes  - lockups under braking (brake bias) and wheelspin on power (diff).
* Track   - straights vs corners -> low-drag vs high-downforce direction.

Cornering is detected from left/right wheel-load asymmetry (reliable and gives
the loaded/outer side) rather than steering units, which vary by car.
Wheel order is [FL, FR, RL, RR] throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .knowledge import (CAMBER_LOADED_IDEAL_HI, CAMBER_LOADED_IDEAL_LO,
                        PRESSURE_HOT_IDEAL_HI, PRESSURE_HOT_IDEAL_LO,
                        PRESSURE_HOT_SANE_HI, PRESSURE_HOT_SANE_LO,
                        TYRE_TEMP_COLD, TYRE_TEMP_HOT)
from .shared_memory import PhysicsSnapshot

_RAD2DEG = 180.0 / math.pi


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _driving(samples: list[PhysicsSnapshot]) -> list[PhysicsSnapshot]:
    return [s for s in samples if s.speed_kmh > 20.0] or samples


# --------------------------------------------------------------------------- #
# Camber
# --------------------------------------------------------------------------- #
@dataclass
class CamberReport:
    front_loaded_camber_deg: float
    rear_loaded_camber_deg: float
    front_advice: str   # "add" | "reduce" | "ok"
    rear_advice: str
    note: str = ""

    def describe(self) -> str:
        return (f"Dynamic camber under load: front {self.front_loaded_camber_deg:+.1f}deg, "
                f"rear {self.rear_loaded_camber_deg:+.1f}deg. {self.note}")

    @property
    def has_issue(self) -> bool:
        return self.front_advice != "ok" or self.rear_advice != "ok"


# Ideal dynamic camber on the loaded tyre at max cornering (degrees, negative).
_CAMBER_IDEAL_LO, _CAMBER_IDEAL_HI = CAMBER_LOADED_IDEAL_LO, CAMBER_LOADED_IDEAL_HI


def analyze_camber(samples: list[PhysicsSnapshot]) -> CamberReport:
    src = _driving(samples)
    # Cornering = clear left/right load asymmetry. The loaded side is the outer.
    front_cambers: list[float] = []
    rear_cambers: list[float] = []
    for s in src:
        fl, fr, rl, rr = s.wheel_load
        left, right = fl + rl, fr + rr
        total = left + right
        if total <= 0:
            continue
        asym = abs(left - right) / total
        if asym < 0.12:      # not cornering hard enough to judge camber
            continue
        # Loaded (outer) tyres are the heavier side.
        if left > right:
            front_cambers.append(s.camber_rad[0] * _RAD2DEG)  # FL
            rear_cambers.append(s.camber_rad[2] * _RAD2DEG)   # RL
        else:
            front_cambers.append(s.camber_rad[1] * _RAD2DEG)  # FR
            rear_cambers.append(s.camber_rad[3] * _RAD2DEG)   # RR

    front = _mean(front_cambers)
    rear = _mean(rear_cambers)

    def advice(cam: float, n: int) -> str:
        if n < 10:
            return "ok"   # not enough cornering data to judge
        if cam > _CAMBER_IDEAL_HI:
            return "add"      # too little (tyre rolling onto outer edge)
        if cam < _CAMBER_IDEAL_LO:
            return "reduce"   # too much
        return "ok"

    fa = advice(front, len(front_cambers))
    ra = advice(rear, len(rear_cambers))
    notes = []
    if fa == "add":
        notes.append("front tyres roll onto their outer edge under load - ADD front camber")
    elif fa == "reduce":
        notes.append("too much front camber (riding the inner edge) - REDUCE it")
    if ra == "add":
        notes.append("ADD rear camber for more rear grip under load")
    elif ra == "reduce":
        notes.append("too much rear camber - REDUCE it")
    note = "; ".join(notes) if notes else "camber looks about right for the grip you're getting."
    return CamberReport(round(front, 1), round(rear, 1), fa, ra, note)


# --------------------------------------------------------------------------- #
# Tyre pressures & temperatures
# --------------------------------------------------------------------------- #
@dataclass
class PressureReport:
    """Hot running pressures (PSI) vs the racing window, per axle.

    AC publishes live tyre pressure (wheelsPressure, PSI). We average the hot
    running pressure per corner and compare to the ~26-28 psi racing window.
    advice is "raise" | "lower" | "ok" per axle; we only advise a change when a
    pressure is outside a wide *sane* band, since the ideal window varies by car.
    """
    front_psi: float
    rear_psi: float
    front_advice: str   # "raise" | "lower" | "ok"
    rear_advice: str
    hot_front: bool     # front tyres overheating (core temp)
    hot_rear: bool
    cold: bool          # tyres never reaching temperature
    note: str = ""

    def describe(self) -> str:
        return (f"Tyre pressures (hot): front {self.front_psi:.1f} psi, "
                f"rear {self.rear_psi:.1f} psi (window "
                f"{PRESSURE_HOT_IDEAL_LO:.0f}-{PRESSURE_HOT_IDEAL_HI:.0f}). "
                f"{self.note}")

    @property
    def has_issue(self) -> bool:
        return (self.front_advice != "ok" or self.rear_advice != "ok"
                or self.cold or self.hot_front or self.hot_rear)


def analyze_pressures(samples: list[PhysicsSnapshot]) -> PressureReport:
    """Average hot pressures/temps per axle and advise toward the racing window."""
    src = _driving(samples)
    if not src:
        return PressureReport(0.0, 0.0, "ok", "ok", False, False, False,
                              "no running data.")

    def axle(idx_a: int, idx_b: int, field_getter) -> float:
        return _mean([(field_getter(s)[idx_a] + field_getter(s)[idx_b]) / 2
                      for s in src])

    front_psi = axle(0, 1, lambda s: s.tyre_pressure)
    rear_psi = axle(2, 3, lambda s: s.tyre_pressure)
    front_temp = axle(0, 1, lambda s: s.tyre_core_temp)
    rear_temp = axle(2, 3, lambda s: s.tyre_core_temp)

    def advice(psi: float) -> str:
        # Only act when clearly outside a sane racing band; inside it, leave
        # pressure to the balance levers (car-specific optimum varies).
        if psi < PRESSURE_HOT_SANE_LO:
            return "raise"
        if psi > PRESSURE_HOT_SANE_HI:
            return "lower"
        return "ok"

    fa, ra = advice(front_psi), advice(rear_psi)
    hot_front = front_temp > TYRE_TEMP_HOT
    hot_rear = rear_temp > TYRE_TEMP_HOT
    cold = max(front_temp, rear_temp) < TYRE_TEMP_COLD

    notes: list[str] = []
    if fa == "raise":
        notes.append(f"front pressure low ({front_psi:.1f} psi) - raise it into the window")
    elif fa == "lower":
        notes.append(f"front pressure high ({front_psi:.1f} psi) - lower it into the window")
    if ra == "raise":
        notes.append(f"rear pressure low ({rear_psi:.1f} psi) - raise it")
    elif ra == "lower":
        notes.append(f"rear pressure high ({rear_psi:.1f} psi) - lower it")
    if cold:
        notes.append("tyres never reach temperature - lower pressures / open ducts less, "
                     "or they simply need more push")
    elif hot_front and hot_rear:
        notes.append("all four tyres overheating - ease the pace or address camber/pressures")
    elif hot_front:
        notes.append("front tyres overheating - front axle is overworked")
    elif hot_rear:
        notes.append("rear tyres overheating - rear axle is overworked")
    note = "; ".join(notes) if notes else "pressures and temps look reasonable."
    return PressureReport(round(front_psi, 1), round(rear_psi, 1), fa, ra,
                          hot_front, hot_rear, cold, note)


# --------------------------------------------------------------------------- #
# Kerbs / suspension
# --------------------------------------------------------------------------- #
@dataclass
class KerbReport:
    bottoming_frac: float
    light_frac: float
    worst_wheel: str
    issue: str          # "bottoming" | "wheels_light" | ""
    note: str = ""

    def describe(self) -> str:
        return (f"Kerbs/suspension: bottoming {self.bottoming_frac*100:.0f}% of the "
                f"time, wheels going light {self.light_frac*100:.0f}%. {self.note}")

    @property
    def has_issue(self) -> bool:
        return bool(self.issue)


_WHEELS = ("front-left", "front-right", "rear-left", "rear-right")


def analyze_kerbs(samples: list[PhysicsSnapshot],
                  susp_max: tuple[float, float, float, float] | None) -> KerbReport:
    src = _driving(samples)
    if not src:
        return KerbReport(0.0, 0.0, "", "")

    # Bottoming: suspension travel near its max (absolute if we know the max,
    # else relative to the top of the observed range).
    bottom_counts = [0, 0, 0, 0]
    light_counts = [0, 0, 0, 0]
    # Per-wheel median load to define "light".
    med_load = [
        _median([s.wheel_load[w] for s in src]) for w in range(4)
    ]
    # Per-wheel bottoming threshold.
    if susp_max and all(m > 1e-4 for m in susp_max):
        bottom_thr = [0.95 * m for m in susp_max]
    else:
        # relative: top 3% of observed travel per wheel
        bottom_thr = []
        for w in range(4):
            trav = sorted(s.suspension_travel[w] for s in src)
            bottom_thr.append(trav[int(len(trav) * 0.97)] if trav else 1e9)

    for s in src:
        for w in range(4):
            if s.suspension_travel[w] >= bottom_thr[w]:
                bottom_counts[w] += 1
            if med_load[w] > 0 and s.wheel_load[w] < 0.12 * med_load[w]:
                light_counts[w] += 1

    n = len(src)
    bottoming_frac = max(bottom_counts) / n
    light_frac = max(light_counts) / n

    issue, note = "", ""
    if bottoming_frac > 0.03:
        worst = _WHEELS[bottom_counts.index(max(bottom_counts))]
        issue = "bottoming"
        note = (f"the {worst} bottoms out (likely over kerbs/compressions) - raise "
                "ride height or stiffen bump stops / bump damping on that end")
        worst_wheel = worst
    elif light_frac > 0.06:
        worst = _WHEELS[light_counts.index(max(light_counts))]
        issue = "wheels_light"
        note = (f"the {worst} goes light over kerbs (loses grip) - soften springs / "
                "bump damping on that end so the wheel follows the kerb")
        worst_wheel = worst
    else:
        worst_wheel = ""
        note = "suspension handles the kerbs cleanly."

    return KerbReport(round(bottoming_frac, 3), round(light_frac, 3),
                      worst_wheel, issue, note)


# --------------------------------------------------------------------------- #
# Brakes / differential
# --------------------------------------------------------------------------- #
@dataclass
class BrakeDiffReport:
    front_lock: bool
    rear_lock: bool
    wheelspin: bool
    note: str = ""

    def describe(self) -> str:
        return f"Braking & traction: {self.note}"

    @property
    def has_issue(self) -> bool:
        return self.front_lock or self.rear_lock or self.wheelspin


def analyze_brakes_diff(samples: list[PhysicsSnapshot]) -> BrakeDiffReport:
    src = _driving(samples)
    braking = [s for s in src if s.brake > 0.5]
    on_power = [s for s in src if s.gas > 0.8 and s.brake < 0.05]

    front_lock = rear_lock = wheelspin = False
    notes = []
    if len(braking) > 10:
        f = _mean([(s.wheel_slip[0] + s.wheel_slip[1]) / 2 for s in braking])
        r = _mean([(s.wheel_slip[2] + s.wheel_slip[3]) / 2 for s in braking])
        if f > r * 1.35 and f > 0.2:
            front_lock = True
            notes.append("fronts lock under braking - shift brake bias REARWARD a little")
        elif r > f * 1.35 and r > 0.2:
            rear_lock = True
            notes.append("rears lock/step out under braking - shift brake bias FORWARD, "
                         "or add diff coast")
    if len(on_power) > 10:
        r = _mean([(s.wheel_slip[2] + s.wheel_slip[3]) / 2 for s in on_power])
        f = _mean([(s.wheel_slip[0] + s.wheel_slip[1]) / 2 for s in on_power])
        if r > f * 1.3 and r > 0.22:
            wheelspin = True
            notes.append("rear wheelspin on power - reduce power-side diff (or add rear grip)")
    note = "; ".join(notes) if notes else "braking and traction look clean."
    return BrakeDiffReport(front_lock, rear_lock, wheelspin, note)


# --------------------------------------------------------------------------- #
# Track character
# --------------------------------------------------------------------------- #
@dataclass
class TrackCharacter:
    avg_speed_kmh: float
    max_speed_kmh: float
    pct_full_throttle: float
    pct_slow: float          # fraction of the lap in slow corners
    kind: str                # "power" | "balanced" | "technical"
    setup_direction: str = ""

    def describe(self) -> str:
        return (f"Track character: {self.kind} (avg {self.avg_speed_kmh:.0f} km/h, "
                f"{self.pct_full_throttle*100:.0f}% full throttle). {self.setup_direction}")


def analyze_track(samples: list[PhysicsSnapshot]) -> TrackCharacter:
    src = _driving(samples)
    if not src:
        return TrackCharacter(0, 0, 0, 0, "balanced")
    speeds = [s.speed_kmh for s in src]
    avg = _mean(speeds)
    mx = max(speeds)
    full = sum(1 for s in src if s.gas > 0.95) / len(src)
    slow = sum(1 for s in src if s.speed_kmh < 0.45 * mx) / len(src)

    if full > 0.55 and slow < 0.22:
        kind = "power"
        direction = ("Favour LOW DRAG: less wing, taller gears - top speed matters "
                     "more than cornering downforce here.")
    elif slow > 0.32 or full < 0.4:
        kind = "technical"
        direction = ("Favour DOWNFORCE and mechanical grip: more wing, softer for "
                     "traction - the corners matter more than top speed here.")
    else:
        kind = "balanced"
        direction = ("Balanced track: trade wing vs top speed to taste; no strong "
                     "bias either way.")
    return TrackCharacter(round(avg, 0), round(mx, 0), round(full, 2),
                          round(slow, 2), kind, direction)
