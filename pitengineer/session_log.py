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

        # Blend the two signals.
        if lap_improved is True or (lap_improved is None and bal_improved):
            improved = True
        elif lap_improved is False and not bal_improved:
            improved = False
        else:
            improved = None

        parts = []
        if lap_delta is not None:
            sign = "-" if lap_delta < 0 else "+"
            parts.append(f"best lap {sign}{abs(lap_delta)/1000:.2f}s")
        parts.append(
            f"balance {'better' if bal_improved else 'similar/worse'} "
            f"({prev.balance_magnitude:.2f}->{cur.balance_magnitude:.2f})"
        )
        verdict_word = (
            "The last change HELPED" if improved
            else "The last change did NOT help" if improved is False
            else "The last change was INCONCLUSIVE"
        )
        text = f"{verdict_word}: {', '.join(parts)} (confidence: {confidence})."
        return Verdict(text, improved, confidence, lap_delta)
