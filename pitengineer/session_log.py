"""Session memory: remember every stint, change, and result per car+track.

This is what lets the auto-tune loop judge "did the last change help?" and get
smarter across sessions. Stored as JSON so it survives restarts and can be
inspected. One record per stint.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .stint import StintReport, fmt_time


def default_history_dir() -> Path:
    d = Path.home() / "Documents" / "Assetto Corsa Setup AI" / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class StintRecord:
    ts: float
    car: str
    track: str
    best_lap_ms: int | None
    median_lap_ms: int | None
    clean_laps: int
    balance_tendency: str
    balance_magnitude: float          # front/rear slip imbalance proxy
    front_rear_temp_delta: float
    lap_spread_ms: int | None = None  # consistency: max-min of clean laps
    consistency: float = 0.5          # 0 erratic .. 1 metronomic
    changes: dict[str, list[int]] = field(default_factory=dict)  # section -> [old, new]

    @staticmethod
    def from_report(car: str, track: str, report: StintReport,
                    changes: dict[str, tuple[int, int]] | None = None) -> "StintRecord":
        s = report.summary
        mag = abs(s.front_slip - s.rear_slip) / max(s.front_slip, s.rear_slip, 1e-6)
        return StintRecord(
            ts=time.time(),
            car=car,
            track=track,
            best_lap_ms=report.metrics.best_lap_ms,
            median_lap_ms=report.metrics.median_lap_ms,
            clean_laps=report.metrics.clean_laps,
            balance_tendency=s.tendency,
            balance_magnitude=round(mag, 4),
            front_rear_temp_delta=round(s.front_temp - s.rear_temp, 1),
            lap_spread_ms=report.metrics.lap_spread_ms,
            consistency=round(report.profile.consistency, 3),
            changes={k: [v[0], v[1]] for k, v in (changes or {}).items()},
        )


@dataclass
class Verdict:
    text: str            # human summary of whether the last change helped
    improved: bool | None  # True/False/None(uncertain)
    confidence: str      # "high" | "medium" | "low"
    lap_delta_ms: int | None


class SessionMemory:
    def __init__(self, history_dir: Path | None = None) -> None:
        self.dir = history_dir or default_history_dir()

    def _path(self, car: str, track: str) -> Path:
        safe = f"{car}__{track}".replace("/", "_").replace("\\", "_")
        return self.dir / f"{safe}.json"

    def load(self, car: str, track: str) -> list[StintRecord]:
        p = self._path(car, track)
        if not p.exists():
            return []
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [StintRecord(**r) for r in raw]

    def append(self, record: StintRecord) -> None:
        p = self._path(record.car, record.track)
        records = self.load(record.car, record.track)
        records.append(record)
        p.write_text(
            json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
        )

    def last(self, car: str, track: str) -> StintRecord | None:
        records = self.load(car, track)
        return records[-1] if records else None

    # --- Reference "ghost" lap (best-ever per car/track) + target time ---
    def _ghost_path(self, car: str, track: str):
        safe = f"{car}__{track}".replace("/", "_").replace("\\", "_")
        return self.dir / f"{safe}.ghost.json"

    def load_ghost(self, car: str, track: str) -> dict | None:
        p = self._ghost_path(car, track)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def save_ghost(self, car: str, track: str, reference: dict) -> None:
        self._ghost_path(car, track).write_text(
            json.dumps(reference, indent=2), encoding="utf-8")

    def target_ms(self, car: str, track: str) -> int | None:
        """Best-ever lap for this car/track, in ms (the time to aim for)."""
        g = self.load_ghost(car, track)
        if g and g.get("lap_time_s"):
            return int(g["lap_time_s"] * 1000)
        return None

    def progress(self, car: str, track: str) -> str:
        """A one-line 'are we getting faster?' summary across the session.

        The whole point is lower lap times - this makes the gain visible.
        """
        records = self.load(car, track)
        bests = [r.best_lap_ms for r in records if r.best_lap_ms]
        if not bests:
            return ""
        n = len(records)
        session_best = min(bests)
        delta = session_best - bests[0]
        # Consistency dimension: best typical (median) pace + most consistent stint.
        medians = [r.median_lap_ms for r in records if r.median_lap_ms]
        best_median = min(medians) if medians else None
        best_consistency = max((r.consistency for r in records), default=0.0)
        cons = f"{best_consistency:.2f}" if best_consistency else "-"

        line = f"Session: {n} stint(s) | best lap {fmt_time(session_best)}"
        if n >= 2:
            line += f" ({delta/1000:+.2f}s vs opening)"
        if best_median:
            line += f" | typical race pace {fmt_time(best_median)}"
        line += f" | best consistency {cons}"
        return line

    def compare(self, prev: StintRecord | None, cur: StintRecord) -> Verdict:
        """Did the change between prev and cur help? Blend lap time + balance."""
        if prev is None or not prev.changes:
            return Verdict("First stint - establishing a baseline.", None, "low", None)

        # Confidence from clean-lap count (more laps = more trustworthy).
        laps = min(prev.clean_laps, cur.clean_laps)
        confidence = "high" if laps >= 5 else "medium" if laps >= 3 else "low"

        lap_delta = None
        lap_improved = None
        if prev.best_lap_ms and cur.best_lap_ms:
            lap_delta = cur.best_lap_ms - prev.best_lap_ms
            # Noise floor: ~0.15s unless we have many laps.
            noise = 150 if laps >= 5 else 300
            if lap_delta < -noise:
                lap_improved = True
            elif lap_delta > noise:
                lap_improved = False

        bal_improved = cur.balance_magnitude < prev.balance_magnitude - 0.02

        # Consistency signals: a better median (typical race pace) and a tighter
        # spread (more repeatable) matter as much as the single best lap - a
        # race is won on consistent laps, not one hero lap.
        median_delta = None
        median_improved = None
        if prev.median_lap_ms and cur.median_lap_ms:
            median_delta = cur.median_lap_ms - prev.median_lap_ms
            noise = 150 if laps >= 5 else 300
            if median_delta < -noise:
                median_improved = True
            elif median_delta > noise:
                median_improved = False
        consistency_improved = cur.consistency > prev.consistency + 0.08

        # Blend all signals. Median pace + consistency count alongside best lap.
        positives = sum(1 for x in (lap_improved, median_improved) if x is True) \
            + (1 if bal_improved else 0) + (1 if consistency_improved else 0)
        negatives = sum(1 for x in (lap_improved, median_improved) if x is False)
        if positives > negatives and positives > 0:
            improved = True
        elif negatives > positives:
            improved = False
        else:
            improved = None

        parts = []
        if lap_delta is not None:
            parts.append(f"best {'-' if lap_delta < 0 else '+'}{abs(lap_delta)/1000:.2f}s")
        if median_delta is not None:
            parts.append(f"median {'-' if median_delta < 0 else '+'}{abs(median_delta)/1000:.2f}s")
        parts.append(f"consistency {prev.consistency:.2f}->{cur.consistency:.2f}")
        parts.append(f"balance {'better' if bal_improved else 'similar'}")
        verdict_word = (
            "The last change HELPED" if improved
            else "The last change did NOT help" if improved is False
            else "The last change was INCONCLUSIVE"
        )
        text = f"{verdict_word}: {', '.join(parts)} (confidence: {confidence})."
        return Verdict(text, improved, confidence, lap_delta)
