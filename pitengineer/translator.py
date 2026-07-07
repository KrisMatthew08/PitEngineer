"""The AI layer: driver complaint -> validated setup changes with reasoning.

We use Claude with a single tool (`propose_setup_changes`) so the model returns
a machine-valid list of changes rather than prose we have to parse. Every
proposed index is then validated and clamped against the car manifest by our
own code — the model advises, the orchestrator guarantees legality.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engines import Engine, make_engine
from .knowledge import format_for_prompt, relevant_entries
from .manifest import CarManifest
from .setup_file import Setup

# JSON schema for the structured response, shared by both AI engines (Claude
# tool-use input_schema / Ollama structured-output format).
CHANGES_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "One or two sentences: what is happening and why.",
        },
        "changes": {
            "type": "array",
            "description": "Ranked changes, most impactful first.",
            "items": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "The parameter section name, e.g. ARB_R.",
                    },
                    "proposed_index": {
                        "type": "integer",
                        "description": "New VALUE index, within [min, max] on a legal step.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this change helps the specific complaint.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["section", "proposed_index", "reason", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["diagnosis", "changes"],
    "additionalProperties": False,
}


@dataclass
class Change:
    section: str
    label: str
    current_index: int
    proposed_index: int
    reason: str
    confidence: str
    clamped: bool  # True if we had to snap the model's value into legal range

    def human_current(self, manifest: CarManifest) -> str:
        p = manifest.get(self.section)
        return p.human(self.current_index) if p else str(self.current_index)

    def human_proposed(self, manifest: CarManifest) -> str:
        p = manifest.get(self.section)
        return p.human(self.proposed_index) if p else str(self.proposed_index)


@dataclass
class Diagnosis:
    text: str
    changes: list[Change]


def _build_setup_context(setup: Setup, manifest: CarManifest) -> str:
    """Describe the current setup + legal ranges, restricted to adjustable params."""
    lines: list[str] = []
    for name, p in manifest.parameters.items():
        cur = setup.get(name)
        cur_str = "unknown" if cur is None else str(cur)
        lines.append(
            f"- {name} ({p.label}): current index={cur_str}, "
            f"range [{p.min}..{p.max}] step {p.step}"
        )
    return "\n".join(lines)


def diagnose(
    complaint: str,
    setup: Setup,
    manifest: CarManifest,
    engine: Engine | None = None,
) -> Diagnosis:
    """Ask the AI engine for changes, then validate every one against the manifest.

    `engine` defaults to a local Ollama model (free). Pass a ClaudeEngine for
    best quality. Either way, the result is validated and clamped here.
    """
    engine = engine or make_engine("ollama")

    grounding = format_for_prompt(relevant_entries(complaint))
    setup_context = _build_setup_context(setup, manifest)

    system = (
        "You are an expert Assetto Corsa race engineer. A driver describes how "
        "the car feels wrong; you translate that into concrete setup changes for "
        "THIS car, tailored to what they describe. Be precise and conservative: "
        "prefer a few high-impact changes over many. Adjust parameters in small, "
        "sensible steps. Only ever propose parameters from the adjustable list, "
        "and keep every proposed_index within the stated range. Explain each "
        "change in terms a driver understands. Respond only with the structured "
        "JSON object (diagnosis + changes)."
    )

    user = (
        f"Car: {manifest.display_name}\n\n"
        f"Driver complaint:\n\"{complaint}\"\n\n"
        f"Vehicle-dynamics grounding (symptom -> likely levers):\n{grounding}\n\n"
        f"Current setup and legal ranges (index space):\n{setup_context}\n\n"
        "Propose the changes."
    )

    result = engine.propose(system, user, CHANGES_SCHEMA)
    return _validate(result, setup, manifest)


def diagnose_autotune(
    report,
    verdict,
    setup: Setup,
    manifest: CarManifest,
    engine: Engine | None = None,
    last_change: dict[str, tuple[int, int]] | None = None,
    segment_context: str = "",
    full_pass: bool = False,
) -> Diagnosis:
    """The auto-tune brain: one iteration of the loop.

    Inputs:
      report      - stint.StintReport (balance + lap metrics + driver profile)
      verdict     - session_log.Verdict on whether the LAST change helped (or None)
      last_change - the exact change applied before this stint {section: (old,new)}

    It personalises to the driver's style, weighs lap-time + balance, reacts to
    whether the last change worked, and proposes the next change(s) - or none if
    the car is dialled in.
    """
    if full_pass:
        # A complete setup at once, built deterministically from the analyzers
        # (reliable + correct direction, not dependent on the small model).
        return _rule_based_full_pass(report, setup, manifest)

    engine = engine or make_engine("ollama")

    # Always include gearing/aero levers alongside the balance-specific ones -
    # lap time often lives in gears and wings, not just the anti-roll bar.
    from .knowledge import LAP_TIME_LEVERS
    entries = relevant_entries(report.summary.tendency) + [LAP_TIME_LEVERS]
    grounding = format_for_prompt(entries)
    setup_context = _build_setup_context(setup, manifest)
    verdict_text = verdict.text if verdict is not None else "This is the first stint (baseline)."
    bias = report.profile.setup_bias()

    # Spell out exactly what was changed last, so revert/keep is grounded.
    if last_change:
        change_lines = "\n".join(
            f"  - {sec}: {old} -> {new}" for sec, (old, new) in last_change.items()
        )
        last_change_text = (
            f"The change you made before this stint was:\n{change_lines}\n"
            "If it did NOT help, move that parameter back toward its old value or "
            "pick a different lever - do not repeat a change that just failed. "
            "If it helped, you may continue in the same direction."
        )
    else:
        last_change_text = "No change was made before this stint (this is a baseline read)."

    _conventions = (
        "INDEX CONVENTIONS (get the direction right): camber is stored as a "
        "NEGATIVE number - a LOWER (more negative) index means MORE camber and "
        "more grip, so to add front grip you DECREASE the camber index. "
        "Anti-roll bars and springs: lower index = softer. Tyre pressure: lower "
        "index = lower pressure. Make sure each proposed_index actually moves in "
        "the direction your reasoning intends. "
        "ONLY use parameters that appear in the adjustable list below - this car "
        "may not have gears, wings, ARBs or dampers; never propose a parameter "
        "that isn't listed. Keep every proposed_index in range. Tailor everything "
        "to the driver's style and aim for CONSISTENT race pace (repeatable lap "
        "after lap, tyres staying in their window) over a one-lap setup. "
        "Follow the vehicle-dynamics grounding directions - they are correct "
        "(e.g. cure understeer by SOFTENING the front anti-roll bar). "
        "Respond only with the structured JSON (diagnosis + changes)."
    )
    if full_pass:
        system = (
            "You are an expert Assetto Corsa race engineer doing a COMPLETE SETUP "
            "PASS for one driver, using all the measured telemetry (balance, "
            "camber, tyre temps/pressures, gearing, aero, suspension/kerbs, "
            "braking, differential, track character, and where time is lost on "
            "the lap). Produce a full, well-rounded setup in ONE go: propose a "
            "change for EVERY area the data shows needs improving - typically 6 "
            "to 12 changes spread across multiple systems (e.g. camber AND "
            "pressures AND gearing AND a damper/brake fix), each a sensible step. "
            "Only leave an area alone if it is genuinely already good. Do not stop "
            "at one or two changes - this is a comprehensive setup, not a single "
            "tweak. " + _conventions
        )
    else:
        system = (
            "You are an expert Assetto Corsa race engineer running an iterative "
            "auto-tune session for one driver. Each stint you get the car's "
            "measured behaviour, the driver's style, and whether your LAST change "
            "helped. Work ONE careful step at a time. Prioritise the change that "
            "gains the most LAP TIME - weigh ALL levers: gearing (rev-limiter / "
            "under-revving) and aero/wings are often bigger gains than an "
            "anti-roll bar tweak. Propose the 2-4 MOST impactful changes this "
            "stint. If the last change did not help, reconsider or revert it. If "
            "the car is fast and balanced and lap times have plateaued, return an "
            "EMPTY changes list and state it is dialled in. " + _conventions
        )

    # If code detected a clear gearing problem, force it to the top so the model
    # can't overlook it - but only steer toward gears the car actually has.
    can_adjust_gears = _has_gear_params(manifest)
    priority = report.gearing.priority_note(can_adjust_gears)
    priority_block = f"{priority}\n\n" if priority else ""
    segment_block = (
        f"WHERE YOU LOSE TIME ON THE LAP:\n{segment_context}\n"
        "Prioritise a setup change that helps the biggest time-loss spot above "
        "(e.g. understeer in a slow corner -> more front grip; slow onto a "
        "straight -> traction/gearing/less drag).\n\n"
        if segment_context else ""
    )

    user = (
        f"Car: {manifest.display_name}\n\n"
        f"{priority_block}"
        f"{segment_block}"
        f"{last_change_text}\n\n"
        f"Result of the last change: {verdict_text}\n\n"
        f"This stint's telemetry:\n{report.describe()}\n\n"
        f"How to bias the setup for THIS driver:\n{bias}\n\n"
        f"Vehicle-dynamics grounding (symptom -> likely levers, follow these "
        f"directions):\n{grounding}\n\n"
        f"Current setup and legal ranges (index space):\n{setup_context}\n\n"
        + ("Do the COMPLETE setup pass now: propose a change for every area above "
           "that needs improving (aim for 6-12 changes across multiple systems)."
           if full_pass else
           "Decide the next step: either propose the next change(s), or return an "
           "empty changes list if it's dialled in.")
    )

    result = engine.propose(system, user, CHANGES_SCHEMA)
    diag = _validate(result, setup, manifest)

    # Guard against a false "dialled in": if the model returns no changes but the
    # telemetry shows a clear, unresolved problem, don't accept it - re-prompt
    # forcefully, then fall back to a rule-based fix so the driver always gets
    # something actionable when the car obviously isn't right.
    if not diag.changes:
        problem = _clear_problem(report, manifest)
        if problem:
            forced = user + (
                f"\n\nYou returned NO changes, but the car is NOT dialled in: "
                f"{problem}. You MUST propose at least one concrete change from the "
                "adjustable list to address this. Do not return an empty list."
            )
            diag = _validate(engine.propose(system, forced, CHANGES_SCHEMA),
                             setup, manifest)
            if not diag.changes:
                diag = _fallback_change(report, setup, manifest, problem)
    return diag


_GEAR_HINTS = ("GEAR", "FINAL", "RATIO")


def _has_gear_params(manifest: CarManifest) -> bool:
    """True if the car exposes any gear-ratio adjustment (read from its setup).
    Some cars (e.g. road cars) simply cannot change gearing."""
    return any(
        any(h in name.upper() for h in _GEAR_HINTS)
        for name in manifest.parameters
    )


def _has_wing(manifest: CarManifest) -> bool:
    return any(name.upper().startswith("WING") for name in manifest.parameters)


def _rule_based_full_pass(report, setup: Setup, manifest: CarManifest) -> Diagnosis:
    """Comprehensive setup pass built from the analyzers - a change for every
    issue the telemetry flags, each in the correct direction, using only the
    car's real parameters. Deterministic and model-independent.
    """
    changes: list[Change] = []
    used: set[str] = set()

    def add(section: str, direction: str, reason: str, steps: int = 1) -> None:
        # direction: "dec"/"inc" (index). Camber "add" = more negative = dec.
        if section in used:
            return
        p = manifest.get(section)
        cur = setup.get(section)
        if p is None or cur is None:
            return
        delta = p.step * steps
        new = p.clamp(cur - delta) if direction == "dec" else p.clamp(cur + delta)
        if new == cur:
            return
        changes.append(Change(section=section, label=p.label, current_index=cur,
                              proposed_index=new, reason=reason,
                              confidence="medium", clamped=False))
        used.add(section)

    s = report.summary
    cam = report.camber

    # 1) Camber (add camber = more negative = DECREASE the index)
    if cam.front_advice == "add":
        for w in ("CAMBER_LF", "CAMBER_RF"):
            add(w, "dec", "add front camber - loaded tyre was rolling onto its edge")
    elif cam.front_advice == "reduce":
        for w in ("CAMBER_LF", "CAMBER_RF"):
            add(w, "inc", "reduce front camber - too much under load")
    if cam.rear_advice == "add":
        for w in ("CAMBER_LR", "CAMBER_RR"):
            add(w, "dec", "add rear camber for more rear grip under load")
    elif cam.rear_advice == "reduce":
        for w in ("CAMBER_LR", "CAMBER_RR"):
            add(w, "inc", "reduce rear camber - too much under load")

    # 2) Tyre temps -> pressures (hotter end: lower pressure a step)
    if s.front_temp - s.rear_temp > 8:
        for w in ("PRESSURE_LF", "PRESSURE_RF"):
            add(w, "dec", "front tyres running hot - drop front pressure")
    elif s.rear_temp - s.front_temp > 8:
        for w in ("PRESSURE_LR", "PRESSURE_RR", "PRESSURE_RL"):
            add(w, "dec", "rear tyres running hot - drop rear pressure")

    # 3) Balance -> anti-roll bars (softer = lower index)
    if s.tendency == "understeer" and s.tendency_strength in ("moderate", "strong"):
        add("ARB_FRONT", "dec", "soften front bar to cut understeer")
        add("ARB_F", "dec", "soften front bar to cut understeer")
    elif s.tendency == "oversteer" and s.tendency_strength in ("moderate", "strong"):
        add("ARB_REAR", "dec", "soften rear bar to cut oversteer")
        add("ARB_R", "dec", "soften rear bar to cut oversteer")

    # 4) Brakes & diff
    b = report.brakes
    if b.front_lock:
        add("FRONT_BIAS", "dec", "fronts lock under braking - shift bias rearward")
        add("BRAKE_BIAS", "dec", "fronts lock under braking - shift bias rearward")
    if b.rear_lock:
        add("FRONT_BIAS", "inc", "rears lock under braking - shift bias forward")
        add("BRAKE_BIAS", "inc", "rears lock under braking - shift bias forward")
        add("DIFF_COAST", "inc", "add coast lock to steady the rear on entry")
    if b.wheelspin:
        add("DIFF_POWER", "dec", "rear wheelspin on power - reduce power diff lock")

    # 5) Kerbs / suspension
    k = report.kerbs
    if k.issue == "bottoming":
        end = "R" if "rear" in k.worst_wheel else "F"
        for w in (f"BUMP_STOP_RATE_L{end}", f"BUMP_STOP_RATE_R{end}"):
            add(w, "inc", "stiffen bump stops - car bottoms out over kerbs")
        for w in (f"ROD_LENGTH_L{end}", f"ROD_LENGTH_R{end}"):
            add(w, "inc", "raise ride height a touch to stop bottoming")
    elif k.issue == "wheels_light":
        end = "R" if "rear" in k.worst_wheel else "F"
        for w in (f"DAMP_BUMP_L{end}", f"DAMP_BUMP_R{end}"):
            add(w, "dec", "soften bump damping so the wheel follows the kerb")

    if not changes:
        return Diagnosis(
            text="Full setup pass: the telemetry doesn't flag a clear problem in "
                 "any area right now - the car looks well sorted. Drive a harder "
                 "or longer stint to surface more.",
            changes=[],
        )
    areas = sorted({c.label.split()[0] for c in changes})
    return Diagnosis(
        text=f"Full setup pass: {len(changes)} changes across {report.track.kind} "
             f"track character, touching {', '.join(areas[:6])}"
             + ("…" if len(areas) > 6 else "") + ".",
        changes=changes,
    )


def _clear_problem(report, manifest: CarManifest | None = None) -> str | None:
    """A short description of an obvious, unresolved problem - or None if fine.

    A gearing issue only counts as actionable if the car can adjust gears, or
    has a wing to trade for straight-line speed.
    """
    s = report.summary
    if s.tendency in ("understeer", "oversteer") and s.tendency_strength in (
            "moderate", "strong"):
        return f"{s.tendency_strength} {s.tendency}"
    delta = s.front_temp - s.rear_temp
    if abs(delta) > 20:
        end = "rear" if delta < 0 else "front"
        return f"{end} tyres overheating ({abs(delta):.0f}C imbalance)"
    if getattr(report, "gearing", None) and report.gearing.issue:
        if manifest is None or _has_gear_params(manifest) or _has_wing(manifest):
            return report.gearing.issue.replace("_", " ")
    return None


def _fallback_change(report, setup: Setup, manifest: CarManifest,
                     problem: str) -> Diagnosis:
    """A guaranteed conservative change when the model won't act.

    Only uses levers the car actually has, and pivots when the obvious lever
    isn't available (gears it can't change -> trim wing for speed).
    """
    s = report.summary
    candidates: list[tuple[str, str]] = []

    # Gearing issue but no gear params -> trim wing for straight-line speed.
    if report.gearing.issue and not _has_gear_params(manifest):
        candidates += [(w, "dec") for w in
                       ("WING_2", "WING_1", "WING_REAR", "WING_9", "WING_3", "WING_10")]
    if s.tendency == "oversteer":
        candidates += [("ARB_REAR", "dec"), ("ARB_R", "dec"),
                       ("PRESSURE_RR", "dec"), ("PRESSURE_RL", "dec"), ("PRESSURE_LR", "dec")]
    elif s.tendency == "understeer":
        candidates += [("ARB_FRONT", "dec"), ("ARB_F", "dec"),
                       ("PRESSURE_LF", "dec"), ("PRESSURE_RF", "dec")]
    if s.rear_temp - s.front_temp > 15:
        candidates += [("PRESSURE_RR", "dec"), ("PRESSURE_RL", "dec"), ("PRESSURE_LR", "dec")]
    elif s.front_temp - s.rear_temp > 15:
        candidates += [("PRESSURE_LF", "dec"), ("PRESSURE_RF", "dec")]

    for sec, dirn in candidates:
        p = manifest.get(sec)
        cur = setup.get(sec)
        if p is None or cur is None:
            continue
        new = p.clamp(cur - p.step) if dirn == "dec" else p.clamp(cur + p.step)
        if new == cur:
            continue
        if sec.upper().startswith("WING"):
            why = "less wing = less drag = more top speed (this car can't change gears)"
        elif "ARB" in sec:
            why = f"softer to reduce {s.tendency}"
        else:
            why = "lower to bring that end's tyre temps down"
        return Diagnosis(
            text=f"The car shows {problem}; here's a conservative step it supports.",
            changes=[Change(
                section=sec, label=p.label, current_index=cur, proposed_index=new,
                reason=f"{p.label}: {why}.", confidence="low", clamped=False,
            )],
        )
    return Diagnosis(
        text=f"The car shows {problem}, but this car has no adjustable lever that "
             "safely addresses it - it may not be fixable via setup here.",
        changes=[],
    )


def diagnose_from_telemetry(
    summary,
    setup: Setup,
    manifest: CarManifest,
    engine: Engine | None = None,
) -> Diagnosis:
    """Diagnose from captured driving telemetry instead of a typed complaint.

    `summary` is a summarizer.TelemetrySummary. We turn its detected tendency
    into knowledge grounding and hand the model the number-backed description.
    """
    engine = engine or make_engine("ollama")

    # Ground the model with the levers for whatever the data shows.
    grounding = format_for_prompt(relevant_entries(summary.tendency))
    setup_context = _build_setup_context(setup, manifest)

    system = (
        "You are an expert Assetto Corsa race engineer analysing live telemetry "
        "from a driver's laps. From the measured data, diagnose the car's "
        "handling problem and propose concrete setup changes for THIS car to fix "
        "it. Be precise and conservative: a few high-impact changes in small, "
        "sensible steps. Only propose parameters from the adjustable list and "
        "keep every proposed_index within the stated range. Explain each change "
        "in terms the driver understands. Respond only with the structured JSON "
        "object (diagnosis + changes)."
    )

    user = (
        f"Car: {manifest.display_name}\n\n"
        f"Telemetry captured from the driver's laps:\n{summary.describe()}\n\n"
        f"Vehicle-dynamics grounding (symptom -> likely levers):\n{grounding}\n\n"
        f"Current setup and legal ranges (index space):\n{setup_context}\n\n"
        "Diagnose from the telemetry and propose the changes."
    )

    result = engine.propose(system, user, CHANGES_SCHEMA)
    return _validate(result, setup, manifest)


def _validate(tool_input: dict, setup: Setup, manifest: CarManifest) -> Diagnosis:
    """Drop changes to unknown params; clamp out-of-range indices into legality."""
    changes: list[Change] = []
    for raw in tool_input.get("changes", []):
        section = raw["section"]
        param = manifest.get(section)
        if param is None:
            # Model hallucinated a non-adjustable parameter — discard it.
            continue

        proposed = int(raw["proposed_index"])
        legal = param.clamp(proposed)
        current = setup.get(section)
        if current is None:
            current = param.clamp(param.min)  # not in file; treat min as baseline

        # Skip no-op changes (model proposed the current value).
        if legal == current:
            continue

        changes.append(
            Change(
                section=section,
                label=param.label,
                current_index=current,
                proposed_index=legal,
                reason=raw["reason"],
                confidence=raw.get("confidence", "medium"),
                clamped=(legal != proposed),
            )
        )

    return Diagnosis(text=tool_input.get("diagnosis", ""), changes=changes)
