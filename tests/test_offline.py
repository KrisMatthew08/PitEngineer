"""Zero-cost offline test: exercises everything except the live AI call.

Run from the project root:

    python -m tests.test_offline

Confirms the setup parser, manifest guardrail (clamping + dropping illegal
changes), and the safe round-trip write all work. No API key required.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ac_setup_ai.manifest import load_manifest
from ac_setup_ai.setup_file import load_setup, write_setup
from ac_setup_ai.translator import _validate

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_setup.ini"
MANIFEST = ROOT / "data" / "manifests" / "generic_gt3.json"


def main() -> int:
    setup = load_setup(SAMPLE)
    manifest = load_manifest(MANIFEST)
    print(f"Loaded {len(setup.values)} sections, {len(manifest.parameters)} adjustable params")

    # A pretend model response: one legal change, one out-of-range (must clamp),
    # one hallucinated parameter (must be dropped), one no-op (must be skipped).
    fake = {
        "diagnosis": "Rear too loose on power exit.",
        "changes": [
            {"section": "ARB_REAR", "proposed_index": 3, "reason": "softer rear bar", "confidence": "high"},
            {"section": "DIFF_POWER", "proposed_index": 999, "reason": "less lock", "confidence": "medium"},
            {"section": "NONEXISTENT", "proposed_index": 1, "reason": "bogus", "confidence": "low"},
            {"section": "WING_REAR", "proposed_index": 6, "reason": "no change", "confidence": "low"},
        ],
    }
    diag = _validate(fake, setup, manifest)
    sections = {c.section for c in diag.changes}

    assert "NONEXISTENT" not in sections, "hallucinated parameter should be dropped"
    assert "WING_REAR" not in sections, "no-op change should be skipped"
    diff_power = next(c for c in diag.changes if c.section == "DIFF_POWER")
    assert diff_power.proposed_index == 100 and diff_power.clamped, "999 should clamp to legal max 100"
    arb = next(c for c in diag.changes if c.section == "ARB_REAR")
    assert arb.proposed_index == 3 and not arb.clamped, "legal value passes through unchanged"
    print("Validation guardrail OK (clamped out-of-range, dropped illegal + no-op)")

    # Round-trip write to a temp copy; confirm changes land and other keys survive.
    tmp = ROOT / "data" / "_tmp_test.ini"
    shutil.copy(SAMPLE, tmp)
    try:
        s2 = load_setup(tmp)
        write_setup(s2, {c.section: c.proposed_index for c in diag.changes}, backup=False)
        s3 = load_setup(tmp)
        assert s3.get("ARB_REAR") == 3
        assert s3.get("DIFF_POWER") == 100
        assert s3.get("FUEL") == 40, "unrelated keys must be preserved byte-for-byte"
        print("Round-trip write OK (changes applied, FUEL preserved)")
    finally:
        tmp.unlink(missing_ok=True)

    print("\nALL OFFLINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
