"""Locate WHERE on the lap you lose time, and what the car is doing there.

Using the normalized track position (0..1) captured with each telemetry sample,
we split the lap into micro-sectors, time each one, and characterise the car's
behaviour in it (corner speed, understeer/oversteer, kerb strikes / bottoming).

Two things come out of this:
* the slowest corners on your best lap (where the lap time lives), and
* per-segment gap to a reference "ghost" lap (best-ever), so you can see where a
  setup change actually found or lost time.

This is what turns "the car understeers" into "you lose 0.4s in the tight left
at 40% of the lap because it understeers there - soften the front".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .shared_memory import PhysicsSnapshot
from .stint import StintData


@dataclass
class Segment:
    index: int
    start_pos: float
    end_pos: float
    time_s: float
    min_speed_kmh: float
    avg_speed_kmh: float
    front_slip: float
    rear_slip: float
    tendency: str            # understeer / oversteer / neutral (in this segment)
    max_susp_travel: float   # peak suspension compression here (kerb/bottoming proxy)
    kind: str                # slow corner / medium corner / fast corner / straight
    gap_s: float | None = None  # vs reference ghost, if available


@dataclass
class SegmentAnalysis:
    n_segments: int
    lap_time_s: float
    segments: list[Segment] = field(default_factory=list)
    worst: list[Segment] = field(default_factory=list)  # biggest time loss
    reference_gap_s: float | None = None                # total gap to ghost

    def describe(self) -> str:
        if not self.segments:
            return "Not enough clean lap data to locate time loss yet."
        lines = [f"Best clean lap this stint: {self.lap_time_s:.3f}s, split into "
                 f"{self.n_segments} micro-sectors."]
        if self.reference_gap_s is not None:
            if self.reference_gap_s < -0.01:
                lines.append(f"That's {abs(self.reference_gap_s):.2f}s FASTER than "
                             "your best-ever lap - new benchmark.")
            elif self.reference_gap_s > 0.01:
                lines.append(f"That's {self.reference_gap_s:.2f}s off your best-ever "
                             "lap - room to find.")
        lines.append("Corners costing you the most (where setup can help):")
        for s in self.worst:
            where = f"{s.start_pos*100:.0f}-{s.end_pos*100:.0f}% of the lap"
            gap = f", {s.gap_s:+.2f}s vs best" if s.gap_s is not None else ""
            behav = s.tendency if s.tendency != "neutral" else "balanced"
            lines.append(
                f"  - {where} ({s.kind}, min {s.min_speed_kmh:.0f} km/h): "
                f"{behav}{gap}."
            )
        return "\n".join(lines)

    def worst_summary(self) -> str:
        """One line for the AI: the biggest setup-actionable problem spot."""
        # Prefer a corner where the car misbehaves; that's what setup fixes.
        actionable = [s for s in self.worst if s.tendency != "neutral"] or self.worst
        if not actionable:
            return ""
        s = actionable[0]
        gap = f" (losing {s.gap_s:.2f}s vs your best there)" if s.gap_s and s.gap_s > 0.02 else ""
        return (f"The car {s.tendency} in the {s.kind} at "
                f"{s.start_pos*100:.0f}% of the lap (min {s.min_speed_kmh:.0f} "
                f"km/h){gap} - a setup change that helps there gains the most time.")


def _complete_laps(data: StintData) -> list[list[int]]:
    """Group sample indices into complete laps (position sweeps ~0 -> ~1)."""
    groups: dict[int, list[int]] = {}
    for i, lap in enumerate(data.laps):
        groups.setdefault(lap, []).append(i)
    complete: list[list[int]] = []
    for lap, idxs in groups.items():
        pos = [data.positions[i] for i in idxs]
        if pos and min(pos) < 0.12 and max(pos) > 0.88 and len(idxs) > 20:
            complete.append(idxs)
    return complete


def _lap_duration(data: StintData, idxs: list[int]) -> float:
    return data.times[idxs[-1]] - data.times[idxs[0]] if len(idxs) > 1 else 0.0


def _classify_corner(min_speed: float) -> str:
    if min_speed < 90:
        return "slow corner"
    if min_speed < 160:
        return "medium corner"
    if min_speed < 220:
        return "fast corner"
    return "straight"


def analyze(data: StintData, n_segments: int = 20,
            reference: dict | None = None) -> SegmentAnalysis:
    """Segment the fastest clean lap and find the biggest time-loss spots.

    `reference` (optional) is a stored ghost: {"lap_time_s", "seg_times": [...]}
    to compute per-segment gap.
    """
    # Need track position to have been captured.
    if not data.positions or max(data.positions) <= 0.0:
        return SegmentAnalysis(n_segments, 0.0)

    laps = _complete_laps(data)
    if not laps:
        return SegmentAnalysis(n_segments, 0.0)

    # Fastest complete lap by wall-clock duration.
    best = min(laps, key=lambda idxs: _lap_duration(data, idxs))
    lap_time = _lap_duration(data, best)
    if lap_time <= 0:
        return SegmentAnalysis(n_segments, 0.0)

    # Bucket the lap's samples into position segments.
    buckets: list[list[int]] = [[] for _ in range(n_segments)]
    for i in best:
        b = min(n_segments - 1, int(data.positions[i] * n_segments))
        buckets[b].append(i)

    segments: list[Segment] = []
    for b, idxs in enumerate(buckets):
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: data.times[i])
        seg_time = data.times[idxs[-1]] - data.times[idxs[0]]
        sp = [data.samples[i].speed_kmh for i in idxs]
        fslip = _mean([(data.samples[i].wheel_slip[0] + data.samples[i].wheel_slip[1]) / 2 for i in idxs])
        rslip = _mean([(data.samples[i].wheel_slip[2] + data.samples[i].wheel_slip[3]) / 2 for i in idxs])
        tendency = _seg_tendency(fslip, rslip)
        max_travel = max(max(data.samples[i].suspension_travel) for i in idxs)
        min_sp = min(sp)
        seg = Segment(
            index=b, start_pos=b / n_segments, end_pos=(b + 1) / n_segments,
            time_s=seg_time, min_speed_kmh=min_sp, avg_speed_kmh=_mean(sp),
            front_slip=fslip, rear_slip=rslip, tendency=tendency,
            max_susp_travel=max_travel, kind=_classify_corner(min_sp),
        )
        if reference and reference.get("seg_times") and b < len(reference["seg_times"]):
            ref_t = reference["seg_times"][b]
            if ref_t and ref_t > 0:
                seg.gap_s = seg_time - ref_t
        segments.append(seg)

    # Rank spots by how ACTIONABLE they are for setup: a corner where the car
    # misbehaves (understeer/oversteer), weighted by how slow it is and how much
    # time you're losing there vs the reference. A neutral straight scores low -
    # setup can't fix it (that's gearing/aero, handled separately).
    def score(s: Segment) -> float:
        issue = 2.0 if s.tendency != "neutral" else 0.0
        corner = 1.0 if s.kind != "straight" else 0.0
        gap = max(0.0, s.gap_s) * 5 if s.gap_s is not None else 0.0
        slow = (300 - s.min_speed_kmh) / 300.0
        return issue + corner + gap + slow

    worst = sorted(segments, key=score, reverse=True)[:3]
    total_gap = None
    if reference and reference.get("lap_time_s"):
        total_gap = lap_time - reference["lap_time_s"]

    return SegmentAnalysis(
        n_segments=n_segments, lap_time_s=lap_time, segments=segments,
        worst=worst, reference_gap_s=total_gap,
    )


def to_reference(analysis: SegmentAnalysis) -> dict:
    """Serialise the fastest lap as a ghost reference to store per car/track."""
    return {
        "lap_time_s": analysis.lap_time_s,
        "seg_times": [s.time_s for s in sorted(analysis.segments, key=lambda x: x.index)],
        "n_segments": analysis.n_segments,
    }


def _seg_tendency(front_slip: float, rear_slip: float) -> str:
    if front_slip <= 0 and rear_slip <= 0:
        return "neutral"
    denom = max(front_slip, rear_slip, 1e-6)
    r = (front_slip - rear_slip) / denom
    if r > 0.10:
        return "understeers"
    if r < -0.10:
        return "oversteers"
    return "neutral"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
