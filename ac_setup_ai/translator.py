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
