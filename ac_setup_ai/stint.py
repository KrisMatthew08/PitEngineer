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

from .driver_profile import DriverProfile, compute_profile
from .shared_memory import ACTelemetry, PhysicsSnapshot
from .summarizer import TelemetrySummary, summarize


@dataclass
class StintData:
    samples: list[PhysicsSnapshot] = field(default_factory=list)
    lap_times_ms: list[int] = field(default_factory=list)  # completed laps this stint


@dataclass
class StintMetrics:
    clean_laps: int
    best_lap_ms: int | None
    median_lap_ms: int | None
    lap_spread_ms: int | None  # max-min of completed laps

    @staticmethod
    def from_laps(lap_times_ms: list[int]) -> "StintMetrics":
        laps = sorted(t for t in lap_times_ms if t and t > 0)
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
        lines.append(self.profile.describe())
        return "\n".join(lines)


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
                last_completed = tele.read_graphics().completed_laps
                while not self._stop.is_set():
                    self.data.samples.append(tele.read_physics())
                    g = tele.read_graphics()
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
    )
