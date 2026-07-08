"""Auto-build a car manifest for ANY car, from the driver's own setup files.

The goal is universality: no hand-written manifest per car. Every AC car the
driver has tuned leaves a trail of setup .ini files under

    Documents/Assetto Corsa/setups/<car_id>/<track>/<name>.ini

Each is a real, valid setup for that exact car. By scanning all of them we learn:
* which parameters the car actually exposes (section names), and
* the range of index values the driver has used for each (safe min/max/step).

That gives a correct, safe manifest for any car or mod, with zero decryption and
zero manual work. (Authoritative full ranges live in the car's packed data.acd;
reading that is a future upgrade - the observed-range approach is safe because
it never proposes a value the driver hasn't already used somewhere.)
"""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

from .manifest import CarManifest, Parameter
from .setup_file import load_setup

# Sections that aren't tunable vehicle parameters - skip them in the manifest.
_SKIP_SECTIONS = {
    "ABOUT", "CAR", "__EXT_PATCH", "TYRES", "FUEL",
}
_SKIP_PREFIXES = ("CUSTOM_SCRIPT_ITEM", "__")

# If a parameter shows only one value across all setups, widen it so the AI has
# room to move. Camber/toe are stored as SIGNED real values (e.g. -29 = -2.9 deg)
# with generous ranges, so give them more absolute room. And widen PROPORTIONALLY
# to the value's magnitude, so large-valued params (anti-roll bars / springs
# stored in N/m, e.g. 60000) get a meaningful window instead of a useless +/-4.
# AC clamps any out-of-range value when it loads the setup, so a generous guess
# is safe. (The authoritative ranges live in the car's data.acd - a future read.)
_SOFT_WIDEN = 4
_WIDEN_CAMBER = 8
_WIDEN_FRAC = 0.12


def default_setups_dir() -> Path:
    """Windows default: Documents/Assetto Corsa/setups."""
    return Path.home() / "Documents" / "Assetto Corsa" / "setups"


def list_cars(setups_dir: Path | None = None) -> list[str]:
    """Car ids the driver has setups for."""
    setups_dir = setups_dir or default_setups_dir()
    if not setups_dir.exists():
        return []
    return sorted(p.name for p in setups_dir.iterdir() if p.is_dir())


def discover_setup_files(car_id: str, setups_dir: Path | None = None) -> list[Path]:
    """All .ini setup files for a car, across every track folder."""
    if not car_id or not car_id.strip():
        return []  # empty car id (e.g. AC not running) - never scan everything
    setups_dir = setups_dir or default_setups_dir()
    car_dir = setups_dir / car_id
    if not car_dir.exists():
        return []
    return sorted(car_dir.rglob("*.ini"))


def _is_adjustable(section: str) -> bool:
    if section in _SKIP_SECTIONS:
        return False
    return not section.startswith(_SKIP_PREFIXES)


def _label_for(section: str) -> str:
    """Human-ish label from a section name, e.g. DAMP_BUMP_LF -> 'Damp bump LF'."""
    corner = {"LF": "FL", "RF": "FR", "LR": "RL", "RR": "RR", "F": "front", "R": "rear"}
    parts = section.split("_")
    words: list[str] = []
    for p in parts:
        words.append(corner.get(p, p.lower()))
    label = " ".join(words)
    return label[:1].upper() + label[1:]


def _group_for(section: str) -> str:
    s = section.upper()
    if s.startswith("PRESSURE"):
        return "tyres"
    if s.startswith("CAMBER") or s.startswith("TOE"):
        return "alignment"
    if s.startswith("ARB"):
        return "arb"
    if s.startswith("SPRING") or s.startswith("ROD_LENGTH") or s.startswith("PACKER"):
        return "springs"
    if s.startswith("DAMP"):
        return "dampers"
    if s.startswith("WING"):
        return "aero"
    if "BIAS" in s or "BRAKE" in s or s.startswith("ABS"):
        return "brakes"
    if s.startswith("DIFF"):
        return "diff"
    return "other"


def _step_from(values: list[int]) -> int:
    """Infer a legal step from the spacing of observed values."""
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return 1
    diffs = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 0]
    step = 0
    for d in diffs:
        step = gcd(step, d)
    return step or 1


def build_manifest_from_setups(
    car_id: str,
    setups_dir: Path | None = None,
    display_name: str | None = None,
) -> CarManifest:
    """Scan a car's setups and derive a manifest (params + safe ranges)."""
    files = discover_setup_files(car_id, setups_dir)
    if not files:
        raise FileNotFoundError(
            f"No setups found for '{car_id}'. Make at least one setup for it "
            "in-game so the app can learn the car's parameters."
        )

    observed: dict[str, list[int]] = {}
    for f in files:
        try:
            setup = load_setup(f)
        except (OSError, ValueError):
            continue
        for section, value in setup.values.items():
            if _is_adjustable(section):
                observed.setdefault(section, []).append(value)

    params: dict[str, Parameter] = {}
    for section, values in observed.items():
        group = _group_for(section)
        lo, hi = min(values), max(values)
        if lo == hi:
            # Only one value ever seen: open a window around it so the AI has
            # room to move. Widen symmetrically. Camber/toe are stored as SIGNED
            # real values (negative), so we must NOT floor at 0 for those - only
            # index-style params (pressures, wing clicks) stay non-negative.
            v = lo
            base = _WIDEN_CAMBER if "CAMBER" in section.upper() else _SOFT_WIDEN
            widen = max(base, round(abs(v) * _WIDEN_FRAC))
            lo, hi = v - widen, v + widen
            if v >= 0:
                lo = max(0, lo)
        if lo > hi:                       # never emit an inverted range
            lo, hi = hi, lo
        params[section] = Parameter(
            name=section,
            label=_label_for(section),
            min=lo,
            max=hi,
            step=_step_from(values),
            group=group,
        )

    return CarManifest(
        car_id=car_id,
        display_name=display_name or car_id,
        parameters=params,
    )


def find_current_setup(
    car_id: str,
    track_id: str,
    setups_dir: Path | None = None,
) -> Path | None:
    """Best guess at the setup the driver is using: most recently modified .ini
    in the car+track folder, falling back to the car's other setups.

    AC doesn't expose the loaded setup's filename via shared memory, so we use
    recency, which reliably matches the setup you were just editing/racing.
    """
    if not car_id or not car_id.strip():
        return None
    setups_dir = setups_dir or default_setups_dir()
    track_dir = setups_dir / car_id / track_id
    candidates: list[Path] = []
    if track_dir.exists():
        candidates = list(track_dir.glob("*.ini"))
    if not candidates:  # fall back to any setup for this car
        candidates = discover_setup_files(car_id, setups_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def track_setup_target(
    car_id: str,
    track_id: str,
    setups_dir: Path | None = None,
    name: str = "pitengineer.ini",
) -> Path | None:
    """Where to SAVE the app's setup for the CURRENT track.

    AC looks for a track's setups under ``setups/<car>/<track>/``. We want the
    written setup to land in the live track's own folder (so AC loads it for
    that track), NOT in ``generic/`` - even when the baseline we read came from
    generic because the track folder had no setup yet. Returns None when we
    don't know the track (caller should fall back to same-folder writing).
    """
    if not car_id or not car_id.strip() or not track_id or not track_id.strip():
        return None
    setups_dir = setups_dir or default_setups_dir()
    return setups_dir / car_id / track_id / name


def save_manifest(manifest: CarManifest, out_path: str | Path) -> Path:
    """Write a derived manifest to JSON (so it can be reused / hand-tuned)."""
    data = {
        "car_id": manifest.car_id,
        "display_name": manifest.display_name,
        "parameters": {
            name: {
                "label": p.label,
                "min": p.min,
                "max": p.max,
                "step": p.step,
                **({"group": p.group} if p.group else {}),
            }
            for name, p in manifest.parameters.items()
        },
    }
    out = Path(out_path)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
