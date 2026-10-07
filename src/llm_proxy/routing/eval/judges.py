"""Judge transports for the evaluation rig — I/O only, evaluation-only.

The request path never reaches a judge over loopback HTTP (ADR-0018): it goes
through the request-free core seam. This transport exists so the rig can measure
a judge before that seam is written, and so an external evaluation script has a
supported way to call one.

Two upstreams are reachable, both plain ``/v1/systemone``:

* Ollama, for local models (``tev1:0.8b``), which is what every latency number in
  ADR-0018 was measured against.
* OpenRouter, which is how the rig measures **Jev** — the reference judge the ADR
  compares its numbers against. Its model id is the floating alias
  ``~typesafe/jev-latest`` (the un-prefixed form 404s).

Every transport returns a :class:`JudgeOutcome` and never raises: a judge that
fails is a judge that abstains, exactly as on the request path.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx2

from llm_proxy.routing.judge.rubric import (
    JudgeVerdict,
    build_judge_questions,
    parse_judge_answers,
)

#: Defaults per transport. The model name is upstream-specific on purpose: the
#: rig talks HTTP directly, so nothing maps an internal model name for it.
OLLAMA_JUDGE_URL = "http://localhost:11434"
OLLAMA_JUDGE_MODEL = "tev1:0.8b"
OPENROUTER_JUDGE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_JUDGE_MODEL = "~typesafe/jev-latest"


@dataclass
class JudgeOutcome:
    """One judge call: what came back, how long it took, and how it failed."""

    verdict: JudgeVerdict | None = None
    latency_ms: float = 0.0
    error: str | None = None
    judge_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        """True when no tier was named, for any reason including failure."""
        return self.verdict is None or self.verdict.abstained


class JudgeTransport(Protocol):
    """What the rig needs from a judge: a name, a model and one call."""

    name: str
    model: str

    async def evaluate(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]] | None = None,
    ) -> JudgeOutcome: ...

    async def aclose(self) -> None: ...


class SystemOneHttpJudge:
    """System One over any HTTP ``/v1/systemone`` endpoint.

    The two upstreams the rig uses share one wire format, so everything except
    the URL and the credential lives here. Subclasses name themselves and fill in
    their own defaults.
    """

    name = "systemone-http"

    def __init__(
        self,
        *,
        url: str,
        model: str,
        timeout: float = 3.0,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self._timeout = timeout
        self._api_key = api_key
        self._client: httpx2.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _get_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=self._timeout)
        return self._client

    async def evaluate(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]] | None = None,
    ) -> JudgeOutcome:
        """Call the judge once; a failure is an abstention, never an exception."""
        body = {
            "model": self.model,
            "state": state,
            "questions": questions or build_judge_questions(),
        }
        started = time.perf_counter()
        try:
            client = await self._get_client()
            response = await client.post(self.url, json=body, headers=self._headers())
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - a failed judge abstains
            return JudgeOutcome(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{exc.__class__.__name__}: {exc}",
                judge_model=self.model,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = data.get("usage") if isinstance(data, dict) else None
        return JudgeOutcome(
            verdict=parse_judge_answers(data.get("answers") if isinstance(data, dict) else None),
            latency_ms=latency_ms,
            judge_model=str(data.get("model")) if isinstance(data, dict) else None,
            input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
            raw=data if isinstance(data, dict) else {},
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if one was opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OllamaSystemOneJudge(SystemOneHttpJudge):
    """System One over Ollama's local ``/v1/systemone`` endpoint.

    Ollama serves ``/v1/systemone`` from locally pulled models only — cloud
    models are rejected upstream — so ``model`` must be a model present in the
    local daemon (``ollama pull tev1:0.8b``).
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = OLLAMA_JUDGE_URL,
        model: str = OLLAMA_JUDGE_MODEL,
        timeout: float = 3.0,
    ) -> None:
        super().__init__(
            url=f"{base_url.rstrip('/')}/v1/systemone",
            model=model,
            timeout=timeout,
        )


class OpenRouterSystemOneJudge(SystemOneHttpJudge):
    """System One over OpenRouter — the Jev reference the ADR is measured against.

    Needs a credential: ``api_key``, or ``OPENROUTER_API_KEY`` in the environment.
    The default model is OpenRouter's floating alias for the latest Jev; the
    un-prefixed ``typesafe/jev-latest`` is not a resolvable id.
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        base_url: str = OPENROUTER_JUDGE_URL,
        model: str = OPENROUTER_JUDGE_MODEL,
        timeout: float = 3.0,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenRouter judge needs a credential: set OPENROUTER_API_KEY or "
                "pass --judge-api-key."
            )
        super().__init__(
            url=f"{base_url.rstrip('/')}/systemone",
            model=model,
            timeout=timeout,
            api_key=key,
        )


__all__ = [
    "OLLAMA_JUDGE_MODEL",
    "OLLAMA_JUDGE_URL",
    "OPENROUTER_JUDGE_MODEL",
    "OPENROUTER_JUDGE_URL",
    "JudgeOutcome",
    "JudgeTransport",
    "OllamaSystemOneJudge",
    "OpenRouterSystemOneJudge",
    "SystemOneHttpJudge",
]
