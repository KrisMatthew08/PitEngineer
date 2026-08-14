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
import os
from math import gcd
from pathlib import Path

from .manifest import CarManifest, Parameter
from .setup_file import load_setup
from .acd import read_setup_ini

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
# Physics-aware widening fractions for parameters with known typical AC ranges.
# When a car's data.acd is encrypted and we only have a narrow observed range,
# we widen symmetrically by this fraction of the observed centre value so the
# AI has meaningful room to move. AC clamps any out-of-range value on load.
_WIDEN_BY_TYPE = [
    # (name_fragment,          widen_frac, floor_at_zero)
    ("ARB",                    0.8,        True),
    ("SPRING_RATE",            0.5,        True),
    ("SPRING",                 0.5,        True),
    ("DAMP",                   0.6,        True),
    ("BUMPSTOP",               0.6,        True),
    ("BUMP_STOP",              0.6,        True),
    ("PRESSURE",               0.25,       True),
    ("CAMBER",                 0.0,        False),  # handled separately
    ("TOE",                    0.5,        False),
    ("ROD_LENGTH",             0.4,        False),
    ("DIFF",                   0.5,        True),
    ("WING",                   0.5,        True),
    ("BRAKE",                  0.3,        True),
    ("GEAR",                   0.3,        True),
]


def default_setups_dir() -> Path:
    """Windows default: Documents/Assetto Corsa/setups."""
    return Path.home() / "Documents" / "Assetto Corsa" / "setups"


def configured_setups_dir(value: str | Path | None = None) -> Path:
    """User-configured setups root, falling back to the Windows default.

    The environment variable lets users point PitEngineer at redirected
    Documents folders such as OneDrive-backed AC setups.
    """
    raw = value if value is not None else os.environ.get("PITENGINEER_SETUPS_DIR")
    if raw:
        return Path(raw).expanduser()
    return default_setups_dir()


def list_cars(setups_dir: Path | None = None) -> list[str]:
    """Car ids the driver has setups for."""
    setups_dir = setups_dir or configured_setups_dir()
    if not setups_dir.exists():
        return []
    return sorted(p.name for p in setups_dir.iterdir() if p.is_dir())


def discover_setup_files(car_id: str, setups_dir: Path | None = None) -> list[Path]:
    """All .ini setup files for a car, across every track folder."""
    if not car_id or not car_id.strip():
        return []  # empty car id (e.g. AC not running) - never scan everything
    setups_dir = setups_dir or configured_setups_dir()
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


def _widen_for(section: str, v: int) -> tuple[int, int]:
    """Return a widened (lo, hi) window for a single-valued parameter.

    Uses physics-aware fractions for parameter types that have known AC ranges,
    falling back to the legacy proportional heuristic for unknowns.
    """
    su = section.upper()
    if "CAMBER" in su:
        # Camber is negative (e.g. -30 = -3.0 deg); always widen generously.
        widen = max(_WIDEN_CAMBER, round(abs(v) * 0.5))
        return v - widen, v + widen

    for fragment, frac, floor_zero in _WIDEN_BY_TYPE:
        if fragment in su:
            widen = max(_SOFT_WIDEN, round(abs(v) * frac))
            lo, hi = v - widen, v + widen
            if floor_zero:
                lo = max(0, lo)
            return lo, hi

    # Generic fallback
    widen = max(_SOFT_WIDEN, round(abs(v) * _WIDEN_FRAC))
    lo, hi = v - widen, v + widen
    if v >= 0:
        lo = max(0, lo)
    return lo, hi


def build_manifest_from_setups(
    car_id: str,
    setups_dir: Path | None = None,
    display_name: str | None = None,
) -> CarManifest:
    """Scan a car's setups and derive a manifest (params + safe ranges)."""
    setups_dir = setups_dir or configured_setups_dir()
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
    
    # Try to get authoritative ranges from setup.ini
    authoritative_setup = read_setup_ini(car_id)

    for section, values in observed.items():
        group = _group_for(section)
        # Check if we have authoritative bounds
        auth_min, auth_max, auth_step = None, None, None
        mapped_min, mapped_max = None, None
        
        if authoritative_setup and authoritative_setup.has_section(section):
            try:
                auth_min = float(authoritative_setup.get(section, 'MIN'))
                auth_max = float(authoritative_setup.get(section, 'MAX'))
                auth_step = float(authoritative_setup.get(section, 'STEP', fallback=1))
                
                # We need to map the physics bounds from setup.ini to the integer space
                # used by the driver setup .ini files (which could be index, value, or scaled value)
                obs_min, obs_max = min(values), max(values)
                S = int(round((auth_max - auth_min) / auth_step)) if auth_step else 0
                
                # Heuristic mapping
                if auth_min <= obs_min and obs_max <= auth_max + 1:
                    mapped_min, mapped_max = int(auth_min), int(auth_max)
                elif auth_min * 10 <= obs_min and obs_max <= auth_max * 10 + 1:
                    mapped_min, mapped_max = int(auth_min * 10), int(auth_max * 10)
                elif auth_min * 100 <= obs_min and obs_max <= auth_max * 100 + 1:
                    mapped_min, mapped_max = int(auth_min * 100), int(auth_max * 100)
                elif 0 <= obs_min and obs_max <= S + 1:
                    mapped_min, mapped_max = 0, S
                else:
                    # If we can't figure out the mapping, we fall back to widened observed bounds
                    pass
            except (ValueError, TypeError):
                pass
        
        if mapped_min is not None and mapped_max is not None:
            lo, hi = mapped_min, mapped_max
            step = _step_from(values)
        else:
            lo, hi = min(values), max(values)
            span = hi - lo
            # If the observed range is very narrow (the driver rarely changes
            # this parameter), widen it using physics-aware heuristics so the
            # AI always has room to move, regardless of how many setups exist.
            v = (lo + hi) // 2
            wlo, whi = _widen_for(section, v)
            # Only widen outward — never shrink an observed range.
            lo = min(lo, wlo)
            hi = max(hi, whi)
            if lo > hi:
                lo, hi = hi, lo
            step = _step_from(values)

        params[section] = Parameter(
            name=section,
            label=_label_for(section),
            min=lo,
            max=hi,
            step=step,
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
    setups_dir = setups_dir or configured_setups_dir()
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
<<<<<<< HEAD
    setups_dir = setups_dir or default_setups_dir()
=======
    setups_dir = setups_dir or configured_setups_dir()
>>>>>>> offline-standalone
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
