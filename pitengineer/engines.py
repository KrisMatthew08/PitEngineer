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


class _ModelLoadError(Exception):
    """Ollama could not load/run the model (usually not enough free memory)."""


class OllamaEngine:
    """Local model via Ollama's /api/chat with structured JSON output.

    If the primary model can't load (e.g. not enough free memory while Assetto
    Corsa is racing), it automatically falls back to a lighter model so the
    stint still gets tuned.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: float = 240.0,
        fallback_model: str | None = "llama3.2:3b",
        keep_alive: str = "10m",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.fallback_model = fallback_model if fallback_model != model else None
        self.keep_alive = keep_alive
        self.name = f"ollama:{model}"
        self.active_model = model  # which model actually answered last

    def propose(self, system: str, user: str, schema: dict) -> dict:
        try:
            self.active_model = self.model
            return self._chat(self.model, system, user, schema)
        except _ModelLoadError as exc:
            if not self.fallback_model:
                raise RuntimeError(
                    f"Ollama couldn't run '{self.model}': {exc}. The model may be "
                    "too big to load while Assetto Corsa is running - close some "
                    "apps to free memory, or use a lighter model."
                ) from exc
            self.active_model = self.fallback_model
            try:
                result = self._chat(self.fallback_model, system, user, schema)
                # Sticky: stay on the lighter model for the rest of the session
                # so we don't waste ~30s retrying the too-big model each stint.
                self.model = self.fallback_model
                self.name = f"ollama:{self.fallback_model} (fell back - low memory)"
                self.fallback_model = None
                return result
            except _ModelLoadError as exc2:
                raise RuntimeError(
                    f"Ollama couldn't run '{self.model}' or fallback "
                    f"'{self.fallback_model}': {exc2}. Close some apps to free "
                    "memory, or pull a smaller model."
                ) from exc2

    def _chat(self, model: str, system: str, user: str, schema: dict) -> dict:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,               # constrain output to this JSON schema
            "think": False,                 # no thinking tokens; we supply grounding
            "keep_alive": self.keep_alive,  # stay warm between stints
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
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            if exc.code >= 500:  # model failed to load / run (usually memory)
                raise _ModelLoadError(detail or f"HTTP {exc.code}") from exc
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
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
