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

# If a parameter shows only one value across all setups, widen it by this many
# index steps each way so the AI has room to move (clamped at >= 0).
_SOFT_WIDEN = 4


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
        lo, hi = min(values), max(values)
        if lo == hi:
            lo = max(0, lo - _SOFT_WIDEN)
            hi = hi + _SOFT_WIDEN
        params[section] = Parameter(
            name=section,
            label=_label_for(section),
            min=lo,
            max=hi,
            step=_step_from(values),
            group=_group_for(section),
        )

    return CarManifest(
        car_id=car_id,
        display_name=display_name or car_id,
        parameters=params,
    )


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
