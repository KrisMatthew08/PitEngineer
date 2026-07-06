"""Pluggable AI backends that turn a prompt + JSON schema into a structured dict.

Two engines, same interface (`propose`):

* OllamaEngine - local, free, no API key. Uses Ollama's structured-output
  `format` field so the model returns JSON matching our schema. Default.
* ClaudeEngine - Anthropic API, best quality, needs a key. Uses tool-use for
  guaranteed structured output.

Either way, the returned dict is validated against the car manifest downstream
(translator._validate) - the engine only has to produce well-formed JSON.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


class Engine(Protocol):
    name: str

    def propose(self, system: str, user: str, schema: dict) -> dict:
        """Return a dict matching `schema` (keys: diagnosis, changes)."""
        ...


class OllamaEngine:
    """Local model via Ollama's /api/chat with structured JSON output."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def propose(self, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,   # Ollama constrains output to this JSON schema
            # Disable "thinking" on reasoning models (qwen3 etc.): we supply our
            # own grounding, and thinking mode both slows CPU inference and can
            # leave the structured output half-filled. Ignored by non-thinking
            # models, so it's safe to always send.
            "think": False,
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? ({exc})"
            ) from exc

        content = data.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON content:\n{content[:500]}"
            ) from exc

    @staticmethod
    def is_running(host: str = "http://localhost:11434") -> bool:
        try:
            with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False


class ClaudeEngine:
    """Anthropic Claude via tool-use for guaranteed structured output."""

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        import anthropic  # lazy: only needed if this engine is chosen

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"

    def propose(self, system: str, user: str, schema: dict) -> dict:
        tool = {
            "name": "propose_setup_changes",
            "description": (
                "Return the ranked list of setup changes that best address the "
                "driver's complaint. Every proposed_index must be within range."
            ),
            "input_schema": schema,
        }
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "propose_setup_changes"},
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "propose_setup_changes":
                return block.input
        raise RuntimeError("Claude did not return a propose_setup_changes tool call.")


def make_engine(kind: str = "ollama", model: str | None = None) -> Engine:
    """Factory: 'ollama' (default) or 'claude'."""
    kind = kind.lower()
    if kind == "ollama":
        return OllamaEngine(model=model or "qwen3:8b")
    if kind == "claude":
        return ClaudeEngine(model=model or "claude-opus-4-8")
    raise ValueError(f"Unknown engine '{kind}'. Use 'ollama' or 'claude'.")
