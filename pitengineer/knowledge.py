"""The engineer's brain: a curated symptom -> cause -> lever map.

This is what makes PitEngineer a reasoner rather than a lookup table. We don't
apply this map directly - we hand it to the model as domain grounding so it
reasons over the specific car, driver, and telemetry while staying anchored to
sound vehicle dynamics.

Lever section names cover BOTH common AC naming conventions (e.g. Kunos GT3s use
ARB_FRONT/ARB_REAR/WING_REAR/BRAKE_BIAS/DIFF_POWER; many mods like RSS use
ARB_F/ARB_R/WING_1/FRONT_BIAS/DIFF_PRELOAD). The model maps these to whichever
parameters the car actually exposes (the real adjustable list is always given in
the prompt). Directions are advisory; the model decides magnitude and our code
validates legality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    symptom: str
    keywords: list[str]
    cause: str
    # (section, direction, why) - direction is "increase" / "decrease" / "either"
    levers: list[tuple[str, str, str]]


KNOWLEDGE_BASE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        symptom="Power-on oversteer (rear steps out on corner exit / throttle)",
        keywords=["oversteer", "loose", "snap", "steps out", "exit", "power",
                  "throttle", "spin", "wag", "tail", "traction"],
        cause="Rear loses grip relative to front under power / lateral load.",
        levers=[
            ("ARB_REAR", "decrease", "Softer rear anti-roll bar increases rear mechanical grip."),
            ("ARB_R", "decrease", "Softer rear anti-roll bar increases rear mechanical grip."),
            ("DIFF_POWER", "decrease", "Less power-side diff lock reduces exit wheelspin/snap."),
            ("DIFF_PRELOAD", "decrease", "Less diff preload frees the rear on power."),
            ("WING_REAR", "increase", "More rear wing adds rear downforce at speed."),
            ("SPRING_LR", "decrease", "Softer rear springs load the rear tyres more."),
            ("SPRING_RR", "decrease", "Softer rear springs load the rear tyres more."),
        ],
    ),
    KnowledgeEntry(
        symptom="Entry / mid understeer (won't turn in / pushes wide)",
        keywords=["understeer", "push", "won't turn", "wont turn", "turn in",
                  "entry", "plough", "plow", "washes out", "front grip", "wide"],
        cause="Front loses grip relative to rear.",
        levers=[
            ("ARB_FRONT", "decrease", "Softer front anti-roll bar increases front mechanical grip."),
            ("ARB_F", "decrease", "Softer front anti-roll bar increases front mechanical grip."),
            ("CAMBER_LF", "increase", "More front negative camber improves front grip when cornering."),
            ("CAMBER_RF", "increase", "More front negative camber improves front grip when cornering."),
            ("WING_FRONT", "increase", "More front wing/splitter adds front downforce if available."),
            ("SPRING_LF", "decrease", "Softer front springs increase front grip."),
            ("TOE_OUT_LF", "increase", "A touch more front toe-out sharpens turn-in."),
        ],
    ),
    KnowledgeEntry(
        symptom="Braking instability (rear nervous / locks under braking)",
        keywords=["braking", "brakes", "under braking", "lock", "unstable",
                  "nervous", "twitchy", "rear lock", "entry snap"],
        cause="Rear axle unloads or over-brakes on entry.",
        levers=[
            ("BRAKE_BIAS", "increase", "Shift brake bias forward for straight-line stability."),
            ("FRONT_BIAS", "increase", "Shift brake bias forward for straight-line stability."),
            ("DIFF_COAST", "increase", "More coast lock stabilises the rear off-throttle."),
            ("ARB_REAR", "decrease", "Softer rear bar keeps the rear tyres loaded."),
            ("ARB_R", "decrease", "Softer rear bar keeps the rear tyres loaded."),
        ],
    ),
    KnowledgeEntry(
        symptom="Tyres overheating or wrong temperature (one end/side)",
        keywords=["overheat", "hot", "cold", "temperature", "temps", "graining",
                  "blister", "greasy", "pressure", "wear"],
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
                  "unpredictable", "darty", "restless", "bumpy"],
        cause="Aero/mechanical balance, dampers, or ride height not settled.",
        levers=[
            ("ARB_FRONT", "either", "Balance front vs rear roll stiffness."),
            ("ARB_REAR", "either", "Balance front vs rear roll stiffness."),
            ("DAMP_BUMP_LF", "either", "Bump damping controls how the front takes load/kerbs."),
            ("DAMP_REBOUND_LR", "either", "Rear rebound controls how the rear settles."),
            ("WING_REAR", "increase", "More rear aero raises high-speed stability."),
            ("HEIGHT_R", "either", "Rake (rear vs front ride height) shifts aero balance."),
            ("ROD_LENGTH_LR", "either", "Rear ride height affects rake and rear grip."),
        ],
    ),
    KnowledgeEntry(
        symptom="Sluggish response / car feels vague and slow to react",
        keywords=["vague", "sluggish", "slow response", "numb", "lazy",
                  "unresponsive", "soft", "floaty"],
        cause="Too soft / too much roll and pitch blunts response.",
        levers=[
            ("ARB_FRONT", "increase", "Stiffer front bar sharpens turn-in response."),
            ("ARB_F", "increase", "Stiffer front bar sharpens turn-in response."),
            ("SPRING_LF", "increase", "Stiffer springs reduce body movement and vagueness."),
            ("SPRING_LR", "increase", "Stiffer springs reduce body movement and vagueness."),
            ("PRESSURE_LF", "increase", "Slightly higher pressure sharpens response."),
        ],
    ),
]


def relevant_entries(complaint: str) -> list[KnowledgeEntry]:
    """Return knowledge entries whose keywords appear in the complaint/tendency.

    Falls back to the full list if nothing matches - better to give the model
    the whole map than starve it of grounding.
    """
    text = complaint.lower()
    hits = [e for e in KNOWLEDGE_BASE if any(k in text for k in e.keywords)]
    return hits or KNOWLEDGE_BASE


def format_for_prompt(entries: list[KnowledgeEntry]) -> str:
    """Render entries as compact grounding text for the model."""
    lines: list[str] = [
        "(Parameter names below are typical examples; map each concept to this "
        "car's actual adjustable parameter listed later in the prompt.)"
    ]
    for e in entries:
        lines.append(f"- Symptom: {e.symptom}")
        lines.append(f"  Cause: {e.cause}")
        for section, direction, why in e.levers:
            lines.append(f"    * {section} ({direction}): {why}")
    return "\n".join(lines)
