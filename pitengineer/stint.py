"""Record and analyse a driving stint.

A stint is a batch of laps between pit visits. We capture physics (for balance
and driver style) and watch the graphics block for lap completions (for lap
times), then boil it down to a StintReport the loop can act on:

* balance (understeer/oversteer + tyre temps) - from summarizer
* driver style - from driver_profile
* lap metrics - best/median lap, consistency, clean-lap count

Capture runs in a background thread so the UI can start/stop it while the driver
laps (CLI: press Enter to stop; GUI: a Stop button).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .analysis import (BrakeDiffReport, CamberReport, KerbReport, TrackCharacter,
                       analyze_brakes_diff, analyze_camber, analyze_kerbs, analyze_track)
from .driver_profile import DriverProfile, compute_profile
from .gearing import GearingReport, analyze_gearing
from .shared_memory import ACTelemetry, PhysicsSnapshot
from .summarizer import TelemetrySummary, summarize


@dataclass
class StintData:
    samples: list[PhysicsSnapshot] = field(default_factory=list)
    lap_times_ms: list[int] = field(default_factory=list)  # completed laps this stint
    car_max_rpm: int = 0  # redline, read from the static block at stint start
    susp_max_travel: tuple[float, float, float, float] | None = None  # from static
    # Parallel per-sample track data (same length/order as `samples`), for
    # locating where on the lap time is lost.
    positions: list[float] = field(default_factory=list)  # 0..1 lap fraction
    times: list[float] = field(default_factory=list)      # monotonic seconds
    laps: list[int] = field(default_factory=list)          # completed-lap index at sample


@dataclass
class StintMetrics:
    clean_laps: int
    best_lap_ms: int | None
    median_lap_ms: int | None
    lap_spread_ms: int | None  # max-min of completed laps

    @staticmethod
    def from_laps(lap_times_ms: list[int]) -> "StintMetrics":
        laps = clean_laps(lap_times_ms)
        if not laps:
            return StintMetrics(0, None, None, None)
        n = len(laps)
        median = laps[n // 2] if n % 2 else (laps[n // 2 - 1] + laps[n // 2]) // 2
        return StintMetrics(
            clean_laps=n,
            best_lap_ms=laps[0],
            median_lap_ms=median,
            lap_spread_ms=laps[-1] - laps[0],
        )


@dataclass
class StintReport:
    metrics: StintMetrics
    summary: TelemetrySummary
    profile: DriverProfile
    gearing: "GearingReport"
    camber: CamberReport
    kerbs: KerbReport
    brakes: BrakeDiffReport
    track: TrackCharacter

    def describe(self) -> str:
        m = self.metrics
        lines = []
        if m.best_lap_ms:
            lines.append(
                f"Laps: {m.clean_laps} clean, best {fmt_time(m.best_lap_ms)}, "
                f"median {fmt_time(m.median_lap_ms)}, "
                f"spread {m.lap_spread_ms/1000:.2f}s."
            )
        else:
            lines.append("No completed laps captured this stint.")
        lines.append(self.summary.describe())
        lines.append(self.gearing.describe())
        lines.append(self.track.describe())
        # Only surface advanced analysis when it found something actionable.
        if self.camber.has_issue:
            lines.append(self.camber.describe())
        if self.kerbs.has_issue:
            lines.append(self.kerbs.describe())
        if self.brakes.has_issue:
            lines.append(self.brakes.describe())
        lines.append(self.profile.describe())
        note = self.consistency_note()
        if note:
            lines.append(note)
        return "\n".join(lines)

    def consistency_note(self) -> str:
        """Attribute poor consistency to the CAR when it's clearly undriveable.

        Low lap-to-lap consistency usually reads as a driver issue - but if the
        tyres are overheating or the car has a strong imbalance, the car is
        making the driver spin/miss, not the other way round. Say so, so the
        engineer fixes the car instead of the app blaming the driver.
        """
        if self.profile.consistency >= 0.35:
            return ""
        s = self.summary
        hottest = max(s.tyre_temp)
        if hottest > 115 or abs(s.front_temp - s.rear_temp) > 20:
            return (
                "Note: your inconsistency is very likely the OVERHEATING tyres "
                "losing grip (easy to spin, especially over kerbs) - that's the "
                "CAR, not your driving. Cooling them / fixing the balance should "
                "make the car easier to drive consistently."
            )
        if s.tendency_strength in ("moderate", "strong"):
            return (
                f"Note: your inconsistency is likely the car's {s.tendency_strength} "
                f"{s.tendency} making it hard to place - fixing the balance should "
                "steady your lap times, so treat this as a CAR problem, not driver "
                "error."
            )
        return ""


def clean_laps(lap_times_ms: list[int], outlier_factor: float = 1.4) -> list[int]:
    """Sorted valid laps, dropping out-laps / in-laps / offs.

    Any lap more than `outlier_factor` x the best is almost certainly not a
    representative flying lap (pit out/in, a spin, going off) - excluding them
    keeps median/spread/consistency meaningful.
    """
    laps = sorted(t for t in lap_times_ms if t and t > 0)
    if not laps:
        return []
    best = laps[0]
    return [lap for lap in laps if lap <= best * outlier_factor]


def fmt_time(ms: int | None) -> str:
    if not ms or ms <= 0:
        return "--:--.---"
    m, rem = divmod(ms, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{m}:{s:02d}.{msec:03d}"


class StintRecorder:
    """Threaded capture of a stint. start() -> drive -> stop() -> analyze()."""

    def __init__(self, rate_hz: float = 20.0) -> None:
        self._rate = rate_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.data = StintData()
        self.error: Exception | None = None

    def start(self) -> None:
        self._stop.clear()
        self.data = StintData()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> StintData:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return self.data

    def _run(self) -> None:
        period = 1.0 / self._rate
        try:
            with ACTelemetry() as tele:
                stat = tele.read_static()
                self.data.car_max_rpm = stat.max_rpm
                self.data.susp_max_travel = stat.suspension_max_travel
                last_completed = tele.read_graphics().completed_laps
                while not self._stop.is_set():
                    g = tele.read_graphics()
                    self.data.samples.append(tele.read_physics())
                    self.data.positions.append(g.car_position)
                    self.data.times.append(time.monotonic())
                    self.data.laps.append(g.completed_laps)
                    if g.completed_laps > last_completed and g.last_time_ms > 0:
                        # A lap just finished; last_time_ms is its time.
                        self.data.lap_times_ms.append(g.last_time_ms)
                        last_completed = g.completed_laps
                    time.sleep(period)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            self.error = exc


def analyze(data: StintData) -> StintReport:
    """Turn captured stint data into a full report."""
    return StintReport(
        metrics=StintMetrics.from_laps(data.lap_times_ms),
        summary=summarize(data.samples),
        profile=compute_profile(data.samples, data.lap_times_ms),
        gearing=analyze_gearing(data.samples, data.car_max_rpm),
        camber=analyze_camber(data.samples),
        kerbs=analyze_kerbs(data.samples, data.susp_max_travel),
        brakes=analyze_brakes_diff(data.samples),
        track=analyze_track(data.samples),
    )
