"""The Translator brain: a curated symptom -> cause -> lever map.

This is what makes the app an AI *reasoner* rather than a lookup table. We don't
apply this map directly — we hand it to the model as domain grounding so it
reasons over the driver's specific car, current setup, and (later) telemetry,
while staying anchored to sound vehicle-dynamics fundamentals.

Each entry names the parameter *sections* (matching manifest / setup keys) most
likely to help, and the direction that usually improves the symptom. Directions
are advisory — the model still decides magnitude and validates against range.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    symptom: str
    keywords: list[str]
    cause: str
    # (section, direction, why) — direction is "increase" / "decrease" / "either"
    levers: list[tuple[str, str, str]]


KNOWLEDGE_BASE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        symptom="Power-on oversteer (rear steps out on corner exit / throttle)",
        keywords=["oversteer", "loose", "snap", "steps out", "exit", "power",
                  "throttle", "spin", "wag", "tail"],
        cause="Rear loses grip relative to front under power/lateral load.",
        levers=[
            ("ARB_REAR", "decrease", "Softer rear anti-roll bar increases rear mechanical grip."),
            ("DIFF_POWER", "decrease", "Less power-side diff lock reduces exit snap."),
            ("WING_REAR", "increase", "More rear wing adds rear downforce at speed."),
            ("PRESSURE_RR", "either", "Tune rear pressures toward the ideal temp window."),
            ("PRESSURE_RL", "either", "Tune rear pressures toward the ideal temp window."),
        ],
    ),
    KnowledgeEntry(
        symptom="Entry understeer (won't turn in / pushes on corner entry)",
        keywords=["understeer", "push", "won't turn", "wont turn", "turn in",
                  "entry", "plough", "plow", "washes out", "front grip"],
        cause="Front loses grip relative to rear on entry.",
        levers=[
            ("ARB_FRONT", "decrease", "Softer front anti-roll bar increases front mechanical grip."),
            ("CAMBER_LF", "increase", "More front negative camber improves front grip in cornering."),
            ("CAMBER_RF", "increase", "More front negative camber improves front grip in cornering."),
            ("BRAKE_BIAS", "decrease", "Shift brake bias rearward to help rotation on entry (small steps)."),
            ("WING_FRONT", "increase", "More front wing/splitter adds front downforce if available."),
        ],
    ),
    KnowledgeEntry(
        symptom="Braking instability (rear nervous / locks under braking)",
        keywords=["braking", "brakes", "under braking", "lock", "unstable",
                  "nervous", "twitchy stop", "rear lock"],
        cause="Rear axle unloads or over-brakes on entry.",
        levers=[
            ("BRAKE_BIAS", "increase", "Shift brake bias forward for straight-line stability."),
            ("DIFF_COAST", "increase", "More coast lock stabilises the rear off-throttle."),
            ("ARB_REAR", "decrease", "Softer rear bar keeps rear tyres loaded."),
            ("PRESSURE_RR", "either", "Correct rear pressures restore contact patch."),
        ],
    ),
    KnowledgeEntry(
        symptom="Tyres overheating or wrong temperature (one end/side)",
        keywords=["overheat", "hot", "cold", "temperature", "temps", "graining",
                  "blister", "greasy", "pressure"],
        cause="Pressure, camber, or load distribution outside the tyre's window.",
        levers=[
            ("PRESSURE_LF", "either", "Lower pressure if over-hot, raise if under-temp."),
            ("PRESSURE_RF", "either", "Lower pressure if over-hot, raise if under-temp."),
            ("PRESSURE_LR", "either", "Lower pressure if over-hot, raise if under-temp."),
            ("PRESSURE_RR", "either", "Lower pressure if over-hot, raise if under-temp."),
            ("CAMBER_LF", "either", "Adjust camber to even inner/outer temps."),
            ("CAMBER_RF", "either", "Adjust camber to even inner/outer temps."),
        ],
    ),
    KnowledgeEntry(
        symptom="Mid-corner instability / nervous overall balance",
        keywords=["mid-corner", "mid corner", "nervous", "unstable", "balance",
                  "unpredictable", "darty", "restless"],
        cause="Aero/mechanical balance or ride height not settled through the corner.",
        levers=[
            ("ARB_FRONT", "either", "Balance front vs rear roll stiffness."),
            ("ARB_REAR", "either", "Balance front vs rear roll stiffness."),
            ("WING_REAR", "increase", "More rear aero raises high-speed stability."),
            ("HEIGHT_R", "either", "Rake (rear vs front ride height) shifts aero balance."),
        ],
    ),
]


def relevant_entries(complaint: str) -> list[KnowledgeEntry]:
    """Return knowledge entries whose keywords appear in the complaint.

    Falls back to the full list if nothing matches — we'd rather give the model
    the whole map than starve it of grounding.
    """
    text = complaint.lower()
    hits = [e for e in KNOWLEDGE_BASE if any(k in text for k in e.keywords)]
    return hits or KNOWLEDGE_BASE


def format_for_prompt(entries: list[KnowledgeEntry]) -> str:
    """Render entries as compact grounding text for the model."""
    lines: list[str] = []
    for e in entries:
        lines.append(f"- Symptom: {e.symptom}")
        lines.append(f"  Cause: {e.cause}")
        for section, direction, why in e.levers:
            lines.append(f"    * {section} ({direction}): {why}")
    return "\n".join(lines)
