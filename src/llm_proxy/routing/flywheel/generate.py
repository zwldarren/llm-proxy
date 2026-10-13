"""Generation leg: one answer per tier-role model through a proxy deployment.

The flywheel replays each prompt through the deployment's OpenAI-compatible
chat endpoint once per role (weak / balanced / strong), non-streaming, so the
pairwise judge compares answers produced under identical conditions.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx2


@dataclass
class GenerationOutcome:
    """One role's answer for one prompt."""

    role: str
    model: str
    text: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


class GenerationClient:
    """OpenAI-compatible ``/v1/chat/completions`` client for the flywheel."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client: httpx2.AsyncClient | None = None

    async def _get_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=self._timeout)
        return self._client

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        role: str,
        reasoning_effort: str | None = None,
    ) -> GenerationOutcome:
        """Generate one answer; a failure is an empty answer plus an error.

        ``reasoning_effort`` rides the OpenAI-compatible field and the proxy's
        thinking layer maps it to the provider's native knob — a same-model
        effort ladder (low/high/max) is a real monotone capability gradient,
        unlike a model trio whose members may not differ where it matters.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.post(self._url, json=body, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - a failed generation is an empty answer
            return GenerationOutcome(
                role=role,
                model=model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = _response_text(data)
        if not text.strip():
            return GenerationOutcome(
                role=role, model=model, latency_ms=latency_ms, error="empty completion"
            )
        return GenerationOutcome(role=role, model=model, text=text, latency_ms=latency_ms)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _response_text(data: Any) -> str:
    """Extract the assistant text from a chat-completions response."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts)
