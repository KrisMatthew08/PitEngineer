"""Read and write Assetto Corsa setup .ini files.

AC setups store each adjustable parameter as its own section with a single
integer ``VALUE=`` key, where the integer is an *index* into that parameter's
allowed range (not the real psi/degree/mm). Example:

    [PRESSURE_LF]
    VALUE=8

We keep everything in index space here; human-readable values are the
manifest's job (see manifest.py).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Setup:
    """A parsed AC setup: an ordered map of section name -> VALUE index.

    We preserve the original raw lines so a round-trip write only touches the
    VALUE lines we actually change and leaves everything else byte-identical.
    """

    path: Path
    values: dict[str, int] = field(default_factory=dict)
    _raw_lines: list[str] = field(default_factory=list, repr=False)

    def get(self, section: str) -> int | None:
        return self.values.get(section)


def load_setup(path: str | Path) -> Setup:
    """Parse a setup .ini into a Setup object."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")  # AC files may carry a BOM
    raw_lines = text.splitlines()

    values: dict[str, int] = {}
    current_section: str | None = None
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
        elif current_section and stripped.upper().startswith("VALUE="):
            raw = stripped.split("=", 1)[1].strip()
            try:
                values[current_section] = int(raw)
            except ValueError:
                # Non-integer VALUE (rare, e.g. tyre compound names) — keep the
                # section known but skip index parsing.
                values[current_section] = _coerce_int(raw)

    return Setup(path=path, values=values, _raw_lines=raw_lines)


def _coerce_int(raw: str) -> int:
    """Best-effort int from a VALUE string; falls back to 0."""
    try:
        return int(float(raw))
    except ValueError:
        return 0


def writable_target(setup_path: str | Path) -> Path:
    """Where to safely WRITE a setup so AC will actually load it.

    AC owns `last.ini` (its auto-saved current setup) and won't reliably reload
    it from disk mid-session - you'd have to restart the game. So if the driver
    is on `last.ini`, we redirect writes to a named `pitengineer.ini` setup they
    can load in the pits, which AC re-reads cleanly on selection.
    """
    p = Path(setup_path)
    if p.stem.lower() in ("last", "last_saved"):
        return p.with_name("pitengineer.ini")
    return p


def write_setup(
    setup: Setup,
    changes: dict[str, int],
    out_path: str | Path | None = None,
    backup: bool = True,
) -> Path:
    """Write ``changes`` (section -> new index) back into the .ini.

    Only the VALUE lines for changed sections are rewritten; all other lines
    are preserved exactly. The original file is backed up to ``<name>.ini.bak``
    before an in-place write unless ``backup=False``.

    Returns the path written.
    """
    target = Path(out_path) if out_path else setup.path

    if backup and target.exists() and out_path is None:
        shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))

    new_lines = list(setup._raw_lines)
    current_section: str | None = None
    applied: set[str] = set()

    for i, line in enumerate(new_lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
        elif (
            current_section in changes
            and stripped.upper().startswith("VALUE=")
        ):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines[i] = f"{indent}VALUE={changes[current_section]}"
            applied.add(current_section)

    # Any change whose section wasn't found in the file is a caller error —
    # surface it rather than silently dropping the change.
    missing = set(changes) - applied
    if missing:
        raise KeyError(
            f"Sections not present in setup file, cannot write: {sorted(missing)}"
        )

    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return target
