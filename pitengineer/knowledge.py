"""The engineer's brain: a curated symptom -> cause -> lever map.

This is what makes PitEngineer a reasoner rather than a lookup table. We don't
apply this map blindly - we hand it to the model as domain grounding so it
reasons over the specific car, driver, and telemetry while staying anchored to
sound vehicle dynamics. The rule-based full-setup pass (translator.py) draws on
the same principles for a model-independent result.

The directions here come from established race-engineering practice: the ACC/AC
setup cheat-sheet (corner entry / mid / exit x understeer / oversteer), the
Driver61 and Trinacria setup guides, and general vehicle dynamics. They apply to
Assetto Corsa too - the physics of anti-roll bars, camber, diff, dampers and
aero are the same across the platforms.

Lever section names cover BOTH common AC naming conventions (e.g. Kunos GT3s use
ARB_FRONT/ARB_REAR/WING_REAR/BRAKE_BIAS/DIFF_POWER; many mods like RSS use
ARB_F/ARB_R/WING_1/FRONT_BIAS/DIFF_PRELOAD). The model maps these to whichever
parameters the car actually exposes (the real adjustable list is always given in
the prompt). Directions are advisory; the model decides magnitude and our code
validates and clamps every value.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Reference target windows (guide-backed), importable by the analyzers so the
# whole app agrees on what "good" looks like.
# --------------------------------------------------------------------------- #

# Hot tyre pressure window (PSI) for racing slick/semi-slick tyres. Driver61's
# ACC guide: GT3 dry 27.3-28.0, GT4 dry 26.6-27.5. We use a slightly wider
# racing window and only ACT on pressures well outside a sane band, because AC
# spans everything from road cars to formula cars.
PRESSURE_HOT_IDEAL_LO = 26.0
PRESSURE_HOT_IDEAL_HI = 28.0
# Outside this band the pressure is almost certainly wrong for any racing tyre.
PRESSURE_HOT_SANE_LO = 24.0
PRESSURE_HOT_SANE_HI = 30.5

# Tyre core temperature window (deg C). Most AC racing tyres like ~75-95 C core;
# above ~105 they are overheating and greasy, below ~65 they never switch on.
TYRE_TEMP_IDEAL_LO = 75.0
TYRE_TEMP_IDEAL_HI = 95.0
TYRE_TEMP_HOT = 105.0
TYRE_TEMP_COLD = 65.0

# Dynamic camber on the LOADED (outer) tyre at max cornering (deg, negative).
# We want the outer contact patch to sit slightly negative under load; if it has
# rolled towards zero/positive the tyre is on its outer edge (add camber), if
# it's very negative it's riding the inner edge (reduce camber).
CAMBER_LOADED_IDEAL_LO = -3.5
CAMBER_LOADED_IDEAL_HI = -1.2


@dataclass
class KnowledgeEntry:
    symptom: str
    keywords: list[str]
    cause: str
    # (section, direction, why) - direction is "increase" / "decrease" / "either"
    # referring to the physical quantity; the `why` text states the setup action
    # in plain terms and the prompt's INDEX CONVENTIONS map it to index space.
    levers: list[tuple[str, str, str]]


KNOWLEDGE_BASE: list[KnowledgeEntry] = [
    # ----------------------------------------------------------------- #
    # UNDERSTEER - front gives up first. Phase tells you the strongest lever:
    # entry = brakes/geometry, mid = roll balance/camber, exit = diff/aero.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Understeer - front won't turn / pushes wide (entry, mid, or exit)",
        keywords=["understeer", "push", "pushes", "won't turn", "wont turn",
                  "turn in", "turn-in", "entry", "plough", "plow", "washes out",
                  "front grip", "wide", "won't rotate", "runs wide", "mid"],
        cause="Front tyres reach their grip limit before the rears. Entry: front "
              "isn't loaded / bar too stiff. Mid: too much front roll or wrong "
              "camber. Exit: diff locking the front line or too little front aero.",
        levers=[
            ("ARB_FRONT", "decrease", "Soften front anti-roll bar - the #1 fix for entry/mid understeer (more front grip in roll)."),
            ("ARB_F", "decrease", "Soften front anti-roll bar - more front mechanical grip through the corner."),
            ("ARB_REAR", "increase", "Alternatively stiffen the rear bar to shift balance toward the front."),
            ("ARB_R", "increase", "Stiffen rear bar to rotate the car more (shifts grip forward)."),
            ("CAMBER_LF", "increase", "More front negative camber - keeps the loaded front tyre flat mid-corner for more grip."),
            ("CAMBER_RF", "increase", "More front negative camber for more mid-corner front grip."),
            ("TOE_OUT_LF", "increase", "A touch more front toe-OUT sharpens turn-in response."),
            ("CASTER_LF", "increase", "More caster adds grip/feel in slow-medium corners (too much hurts fast-corner turn-in)."),
            ("SPRING_LF", "decrease", "Softer front springs load the front tyres more for grip."),
            ("WING_FRONT", "increase", "More front wing/splitter - the key fix for high-speed (mid/exit) understeer."),
            ("BRAKE_BIAS", "decrease", "Shift brake bias slightly rearward to help the car rotate on trail-braking entry."),
            ("FRONT_BIAS", "decrease", "Shift brake bias rearward a touch to cut entry understeer under braking."),
            ("DIFF_PRELOAD", "increase", "More diff preload can tighten a loose entry, but reduce it if the diff is locking the front line on exit."),
            ("PRESSURE_LF", "decrease", "Drop front pressure toward the window to grow the front contact patch."),
            ("HEIGHT_F", "decrease", "Lower front / raise rear ride height (more rake) shifts grip and aero forward."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # OVERSTEER - rear gives up first. Entry = trail-brake/geometry, mid = roll
    # balance, exit = throttle/diff/traction.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Oversteer - rear steps out / snaps (entry, mid, or power-on exit)",
        keywords=["oversteer", "loose", "snap", "snappy", "steps out", "exit",
                  "power", "throttle", "spin", "wag", "tail", "rotates too much",
                  "rear grip", "nervous rear", "kick", "lift-off", "lift off"],
        cause="Rear tyres reach their limit before the fronts. Entry: rear light "
              "under braking / too much rear bar. Mid: rear roll grip low. Exit: "
              "diff snap or rear tyre overload under throttle.",
        levers=[
            ("ARB_REAR", "decrease", "Soften rear anti-roll bar - the #1 fix for mid/exit oversteer (more rear grip in roll)."),
            ("ARB_R", "decrease", "Soften rear bar for more rear mechanical grip."),
            ("ARB_FRONT", "increase", "Alternatively stiffen the front bar to steady the rear."),
            ("ARB_F", "increase", "Stiffen front bar to shift balance away from the loose rear."),
            ("WING_REAR", "increase", "More rear wing - the key fix for high-speed / power-on oversteer (rear downforce)."),
            ("WING_9", "increase", "More rear wing adds rear downforce and high-speed stability."),
            ("DIFF_POWER", "decrease", "Less power-side diff lock cuts exit wheelspin and snap."),
            ("DIFF_PRELOAD", "decrease", "Less diff preload frees the rear off-throttle and reduces mid-corner snap."),
            ("CAMBER_LR", "increase", "More rear negative camber keeps the loaded rear tyre planted under load."),
            ("CAMBER_RR", "increase", "More rear negative camber for more rear grip."),
            ("SPRING_LR", "decrease", "Softer rear springs load the rear tyres more evenly."),
            ("SPRING_RR", "decrease", "Softer rear springs increase rear grip."),
            ("BRAKE_BIAS", "increase", "Shift brake bias forward to stop the rear stepping out on entry."),
            ("FRONT_BIAS", "increase", "Shift brake bias forward for a stable braking entry."),
            ("DIFF_COAST", "increase", "More coast lock stabilises the rear off-throttle into the corner."),
            ("PRESSURE_LR", "decrease", "Drop rear pressure toward the window for a bigger rear contact patch."),
            ("HEIGHT_R", "increase", "Raise rear / lower front ride height a touch to add rear grip (less rake)."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Braking - front lock (entry understeer under braking).
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Front tyres lock under braking (car goes straight on / flat-spots)",
        keywords=["front lock", "fronts lock", "locking", "lock up", "lock-up",
                  "flat spot", "flatspot", "braking", "brakes", "under braking",
                  "won't stop", "abs"],
        cause="Front axle over-braked relative to grip / bias too far forward.",
        levers=[
            ("BRAKE_BIAS", "decrease", "Shift brake bias rearward so the fronts stop locking."),
            ("FRONT_BIAS", "decrease", "Move bias away from the locking front axle."),
            ("BRAKE_POWER", "decrease", "Slightly less total brake power if the fronts lock everywhere (last resort)."),
            ("ABS", "increase", "If the car has adjustable ABS, raise it to stop lock-ups."),
            ("PRESSURE_LF", "decrease", "Bring front pressure into its window for a bigger, more consistent contact patch."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Braking - rear lock / entry instability (rear steps out braking).
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Rear locks / steps out under braking (unstable, nervous entry)",
        keywords=["rear lock", "rears lock", "braking instability", "unstable",
                  "nervous", "twitchy", "entry snap", "rear steps out braking",
                  "rear light", "under braking"],
        cause="Rear axle unloads or over-brakes on entry; bias too far rearward "
              "or coast diff too open.",
        levers=[
            ("BRAKE_BIAS", "increase", "Shift brake bias forward for a stable braking entry."),
            ("FRONT_BIAS", "increase", "Move bias toward the front so the rear stops locking."),
            ("DIFF_COAST", "increase", "More coast lock steadies the rear off-throttle."),
            ("ARB_REAR", "decrease", "Softer rear bar keeps the rear tyres loaded under braking."),
            ("ARB_R", "decrease", "Softer rear bar for a more planted rear on entry."),
            ("TOE_IN_LR", "increase", "A touch of rear toe-in adds straight-line and braking stability."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Traction - wheelspin limiting exit.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Poor traction / wheelspin bogging the corner exit",
        keywords=["traction", "wheelspin", "spin up", "spinning", "bog", "can't "
                  "put power down", "exit", "acceleration", "drive off", "lights up"],
        cause="Rear can't put the power down - diff too aggressive, rear grip low, "
              "or too little rear load/aero on exit.",
        levers=[
            ("DIFF_POWER", "decrease", "Less power-side diff lock lets the rear hook up progressively."),
            ("DIFF_PRELOAD", "decrease", "Less preload smooths power delivery out of slow corners."),
            ("ARB_REAR", "decrease", "Softer rear bar loads both rear tyres for more traction."),
            ("ARB_R", "decrease", "Softer rear bar improves mechanical traction on exit."),
            ("WING_REAR", "increase", "More rear wing adds rear grip for power-down (costs a little top speed)."),
            ("SPRING_LR", "decrease", "Softer rear springs put more rubber down under acceleration."),
            ("HEIGHT_R", "increase", "A touch more rear ride height/rake can help rear squat and traction."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Tyres - temps and pressures.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Tyres overheating / wrong temperature / pressures out of window",
        keywords=["overheat", "overheating", "hot", "cold", "temperature",
                  "temps", "graining", "blister", "greasy", "pressure", "psi",
                  "wear", "degradation", "falling off", "cooking"],
        cause="Pressure or camber outside the tyre's window, or one axle overworked. "
              "Aim for ~26-28 psi hot and ~75-95 C core on racing tyres.",
        levers=[
            ("PRESSURE_LF", "either", "Raise pressure if hot pressure is below the window, lower it if above."),
            ("PRESSURE_RF", "either", "Target ~26-28 psi hot; too low overheats from flex, too high overheats the centre."),
            ("PRESSURE_LR", "either", "Match rear pressures into the window as well."),
            ("PRESSURE_RR", "either", "Match rear pressures into the window as well."),
            ("CAMBER_LF", "either", "Adjust camber to even inner/outer temps (more negative if the outer edge is hot)."),
            ("CAMBER_RF", "either", "Camber controls which part of the tread carries load and heat."),
            ("ARB_FRONT", "either", "If one whole axle overheats, it's overworked - use roll balance to share load."),
            ("ARB_REAR", "either", "Balance roll stiffness so neither axle is doing all the work."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Kerbs / bumps - car unsettled over kerbs and compressions.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Car unsettled over kerbs / bumps (bottoming or skipping)",
        keywords=["kerb", "curb", "bump", "bumpy", "bottoming", "bottom out",
                  "skip", "skipping", "unsettled", "hops", "compression", "rides"],
        cause="Suspension bottoming out (too low/soft on bump) or a wheel going "
              "light and skating over the kerb (too stiff to follow it).",
        levers=[
            ("BUMP_STOP_RATE_LF", "increase", "Stiffen bump stops if the car bottoms out over kerbs/compressions."),
            ("BUMP_STOP_RATE_LR", "increase", "Stiffen bump stops on the bottoming end."),
            ("ROD_LENGTH_LF", "increase", "Raise ride height a little to stop bottoming (front)."),
            ("ROD_LENGTH_LR", "increase", "Raise ride height a little to stop bottoming (rear)."),
            ("DAMP_BUMP_LF", "decrease", "Soften bump damping so the wheel follows the kerb instead of skating."),
            ("DAMP_FAST_BUMP_LF", "decrease", "Lower fast-bump damping to absorb sharp kerb impacts."),
            ("DAMP_REBOUND_LF", "decrease", "Soften rebound so the wheel returns to the road quickly after a kerb."),
        ],
    ),
    # ----------------------------------------------------------------- #
    # Response / overall balance - vague, sluggish, or restless.
    # ----------------------------------------------------------------- #
    KnowledgeEntry(
        symptom="Vague / sluggish response, or restless mid-corner balance",
        keywords=["vague", "sluggish", "slow response", "numb", "lazy",
                  "unresponsive", "soft", "floaty", "mid-corner", "mid corner",
                  "balance", "unpredictable", "darty", "restless"],
        cause="Too much body movement blunts response, or dampers/rake not "
              "settling the platform mid-corner.",
        levers=[
            ("ARB_FRONT", "either", "Balance front vs rear roll stiffness for a settled, responsive platform."),
            ("ARB_REAR", "either", "Balance roll stiffness front to rear."),
            ("SPRING_LF", "increase", "Stiffer springs cut vagueness and sharpen turn-in (if the car floats)."),
            ("DAMP_BUMP_LF", "either", "Bump damping controls how the front takes load and kerbs."),
            ("DAMP_REBOUND_LR", "either", "Rear rebound controls how the rear settles after load changes."),
            ("HEIGHT_R", "either", "Rake (rear vs front ride height) trims aero balance and mid-corner feel."),
        ],
    ),
]


# Lap-time levers (gearing + aero). Always relevant when optimising for pace,
# so the auto-tune loop includes this in the grounding every stint.
LAP_TIME_LEVERS = KnowledgeEntry(
    symptom="Lap time on straights / gearing / aero (not a handling complaint)",
    keywords=["gear", "gearing", "aero", "wing", "top speed", "straight",
              "rev limiter", "redline", "drag", "downforce", "acceleration"],
    cause="Gears mismatched to the track, or wing level trading straight speed "
          "against cornering grip. Match track character: low-drag for power "
          "tracks, more downforce for technical ones.",
    levers=[
        ("GEARSET", "either", "Pick a gearset that reaches redline near the end of the longest straight."),
        ("FINAL_GEAR_RATIO", "either", "Taller final = more top speed; shorter = more acceleration."),
        ("GEAR_1", "either", "Lengthen a gear that hits the limiter too early; shorten one that bogs."),
        ("WING_REAR", "either", "Less rear wing raises top speed (less drag); more adds cornering grip and stability."),
        ("WING_9", "either", "Less wing raises top speed; more adds cornering downforce."),
        ("WING_1", "either", "Front wing: balance front aero against the rear to keep the car neutral at speed."),
        ("WING_FRONT", "either", "Balance front aero with rear to keep the car neutral at speed."),
    ],
)


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
        "car's actual adjustable parameter listed later in the prompt. Directions "
        "describe the SETUP ACTION - follow the plain-language note.)"
    ]
    for e in entries:
        lines.append(f"- Symptom: {e.symptom}")
        lines.append(f"  Cause: {e.cause}")
        for section, direction, why in e.levers:
            lines.append(f"    * {section} ({direction}): {why}")
    return "\n".join(lines)
