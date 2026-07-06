"""Car adjustability manifest: what's tunable, and each parameter's legal range.

The manifest is the guardrail. The AI proposes changes; the orchestrator
validates every proposed index against the manifest's min/max/step before it is
ever shown or written. The model advises, this file's data guarantees legality.

A manifest is a JSON file (see data/manifests/) describing one car:

    {
      "car_id": "ks_ferrari_488_gt3",
      "display_name": "Ferrari 488 GT3",
      "parameters": {
        "PRESSURE_LF": {
          "label": "Front left tyre pressure",
          "min": 0, "max": 30, "step": 1,
          "unit": "psi", "value_at_min": 17.0, "value_per_step": 0.5,
          "group": "tyres"
        },
        ...
      }
    }

``value_at_min`` / ``value_per_step`` are optional and only used to render a
human-readable value (e.g. index 8 -> 21.0 psi). Everything the AI touches is
the integer index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Parameter:
    name: str
    label: str
    min: int
    max: int
    step: int = 1
    unit: str | None = None
    value_at_min: float | None = None
    value_per_step: float | None = None
    group: str | None = None

    def clamp(self, index: int) -> int:
        """Clamp an index into range and snap it to the nearest legal step."""
        index = max(self.min, min(self.max, index))
        # Snap to step relative to min.
        offset = round((index - self.min) / self.step) * self.step
        return self.min + offset

    def is_legal(self, index: int) -> bool:
        return (
            self.min <= index <= self.max
            and (index - self.min) % self.step == 0
        )

    def human(self, index: int) -> str:
        """Render an index as a human-readable value, if we know the mapping."""
        if self.value_at_min is None or self.value_per_step is None:
            return f"index {index}"
        real = self.value_at_min + (index - self.min) * self.value_per_step
        unit = f" {self.unit}" if self.unit else ""
        # Trim trailing .0 for tidiness.
        real_str = f"{real:g}"
        return f"{real_str}{unit} (index {index})"


@dataclass
class CarManifest:
    car_id: str
    display_name: str
    parameters: dict[str, Parameter]

    def get(self, name: str) -> Parameter | None:
        return self.parameters.get(name)

    def adjustable_names(self) -> list[str]:
        return list(self.parameters.keys())


def load_manifest(path: str | Path) -> CarManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    params: dict[str, Parameter] = {}
    for name, spec in data["parameters"].items():
        params[name] = Parameter(name=name, **spec)
    return CarManifest(
        car_id=data["car_id"],
        display_name=data["display_name"],
        parameters=params,
    )
