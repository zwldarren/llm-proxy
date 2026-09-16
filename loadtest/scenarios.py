"""Scenario matrix for llm-proxy load testing.

The proxy is not one performance profile: a request's cost depends on the
client protocol, the provider wire dialect, and — above all — the *conversion
tier* the pipeline picks for that pair (see ``llm_proxy.core.conversion`` and
CONTEXT.md "Conversion plan"):

- **Native passthrough** — the stashed client body and the upstream SSE are
  forwarded verbatim (client protocol == provider dialect).
- **Wire reuse** — the client body is reused after ``model``/``stream`` are
  rewritten (client protocol is wire-compatible with the provider), and the
  response body may ride verbatim too.
- **Full conversion** — canonical parse -> ``InternalRequest`` -> provider
  serializer, and the reverse for the response/stream.

A single "/v1/chat/completions against one openai-compatible upstream" test
only measures the wire-reuse tier. This module enumerates the representative
matrix so each tier, protocol and provider dialect can be loaded independently
or as a weighted mix.

Everything here is data + pure functions, with no locust import, so the seed
script and the fake upstream can share the same topology.
"""

import os
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Protocols (client-facing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolSpec:
    """A client-facing protocol endpoint and how to drive/validate it."""

    key: str
    path: str
    #: Substring that must appear in a streamed body for the stream to count
    #: as complete. ``None`` for non-streaming protocols.
    terminal_marker: str | None
    #: Extra client headers the protocol expects (e.g. anthropic-version).
    headers: dict[str, str] = field(default_factory=dict)
    supports_stream: bool = True


PROTOCOLS: dict[str, ProtocolSpec] = {
    "chat": ProtocolSpec(
        key="chat",
        path="/v1/chat/completions",
        terminal_marker="[DONE]",
    ),
    "messages": ProtocolSpec(
        key="messages",
        path="/v1/messages",
        terminal_marker="message_stop",
        headers={"anthropic-version": "2023-06-01"},
    ),
    "responses": ProtocolSpec(
        key="responses",
        path="/v1/responses",
        terminal_marker="response.completed",
    ),
    "embeddings": ProtocolSpec(
        key="embeddings",
        path="/v1/embeddings",
        terminal_marker=None,
        supports_stream=False,
    ),
}


# ---------------------------------------------------------------------------
# Providers (upstream-facing)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """One upstream provider + its proxy-level model alias.

    ``tiers`` documents which conversion tiers this provider exercises for
    each client protocol, for reporting/selection (the pipeline computes the
    real verdict; this is documentation + test intent).
    """

    key: str
    name: str
    type: str
    base_url: str
    model: str
    provider_model: str
    supports_embeddings: bool = True
    metadata: dict = field(default_factory=dict)
    endpoint_base_urls: dict[str, str] = field(default_factory=dict)
    description: str = ""


def _env_upstream(path: str) -> str:
    """Upstream base URL as seen from inside the proxy container."""
    host = os.getenv("FAKE_UPSTREAM_URL", "http://fake-upstream:8900")
    return f"{host.rstrip('/')}{path}"


PROVIDERS: dict[str, ProviderSpec] = {
    # Chat Completions wire dialect. ``compatible_protocols={"openai"}`` means
    # the openai client protocol rides WIRE_REUSE, and the response stream is
    # forwarded natively (``supports_native_streaming("openai")``).
    "openai_compat": ProviderSpec(
        key="openai_compat",
        name="fake-openai-compat",
        type="openai-compatible",
        base_url=_env_upstream("/v1"),
        model="fake-model",
        provider_model="fake-model",
        description="Chat Completions dialect: chat=wire-reuse + native stream; "
        "messages/responses=full conversion.",
    ),
    # OpenAI Responses dialect. The openai client protocol IS the native
    # dialect here; chat/messages are converted to Responses upstream.
    "openai_native": ProviderSpec(
        key="openai_native",
        name="fake-openai-native",
        type="openai",
        base_url=_env_upstream("/v1"),
        model="fake-openai",
        provider_model="fake-openai",
        description="Responses dialect: responses=native passthrough; "
        "chat/messages=full conversion.",
    ),
    # Anthropic Messages dialect. messages=native passthrough.
    "anthropic_native": ProviderSpec(
        key="anthropic_native",
        name="fake-anthropic",
        type="anthropic",
        base_url=_env_upstream(""),
        model="fake-anthropic",
        provider_model="fake-anthropic",
        supports_embeddings=False,
        description="Anthropic dialect: messages=native passthrough; "
        "chat/responses=full conversion.",
    ),
    # NativePassthroughChatBase: one provider exposing three dialects.
    # messages -> native (root /anthropic/v1/messages), responses -> native
    # (/responses), chat -> wire-reuse request + converted stream (the
    # reasoning-echo veto keeps it off the native stream tier).
    "deepseek_multi": ProviderSpec(
        key="deepseek_multi",
        name="fake-deepseek",
        type="deepseek",
        base_url=_env_upstream("/v1"),
        model="fake-deepseek",
        provider_model="fake-deepseek",
        description="Multi-dialect: messages/responses=native passthrough, "
        "chat=wire-reuse request + converted stream.",
    ),
    # Gemini dialect: every client protocol is a full conversion.
    "gemini": ProviderSpec(
        key="gemini",
        name="fake-gemini",
        type="gemini",
        base_url=_env_upstream("/v1beta"),
        model="fake-gemini",
        provider_model="fake-gemini",
        description="Gemini dialect: full conversion for every protocol.",
    ),
    # Ollama dialect: every client protocol is a full conversion.
    "ollama": ProviderSpec(
        key="ollama",
        name="fake-ollama",
        type="ollama",
        base_url=_env_upstream(""),
        model="fake-ollama",
        provider_model="fake-ollama",
        description="Ollama dialect: full conversion for every protocol.",
    ),
}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

#: Expected conversion tier per (protocol, provider) — documentation and
#: grouping data. Values are one of ``native``, ``wire_reuse``, ``conversion``
#: or a compound string for the request/stream mix.
_EXPECTED_TIERS: dict[tuple[str, str], str] = {
    ("chat", "openai_compat"): "wire_reuse+native_stream",
    ("chat", "openai_native"): "conversion",
    ("chat", "anthropic_native"): "conversion",
    ("chat", "deepseek_multi"): "wire_reuse+conversion",
    ("chat", "gemini"): "conversion",
    ("chat", "ollama"): "conversion",
    ("messages", "openai_compat"): "conversion",
    ("messages", "openai_native"): "conversion",
    ("messages", "anthropic_native"): "native",
    ("messages", "deepseek_multi"): "native",
    ("messages", "gemini"): "conversion",
    ("messages", "ollama"): "conversion",
    ("responses", "openai_compat"): "conversion",
    ("responses", "openai_native"): "native",
    ("responses", "anthropic_native"): "conversion",
    ("responses", "deepseek_multi"): "native",
    ("responses", "gemini"): "conversion",
    ("responses", "ollama"): "conversion",
    ("embeddings", "openai_compat"): "native",
    ("embeddings", "openai_native"): "native",
    ("embeddings", "deepseek_multi"): "native",
    ("embeddings", "gemini"): "conversion",
    ("embeddings", "ollama"): "conversion",
}

#: The curated default mix: at least one representative per protocol and tier.
_CORE: frozenset[str] = frozenset(
    {
        "chat_openai_compat_nonstream",  # wire-reuse request + response
        "chat_openai_compat_stream",  # wire-reuse request + native stream
        "chat_openai_native_nonstream",  # full conversion -> Responses
        "messages_anthropic_native_stream",  # native passthrough
        "messages_openai_compat_stream",  # full conversion -> Chat Completions
        "responses_openai_native_stream",  # native passthrough
        "responses_anthropic_native_nonstream",  # full conversion -> Messages
        "embeddings_openai_compat_nonstream",
        "embeddings_gemini_nonstream",
    }
)


@dataclass(frozen=True)
class Scenario:
    """One (protocol, provider, stream) point in the load matrix."""

    key: str
    protocol: str
    provider: str
    stream: bool = False
    weight: int = 1
    expected_tier: str = ""
    groups: tuple[str, ...] = ()
    description: str = ""

    @property
    def path(self) -> str:
        return PROTOCOLS[self.protocol].path

    @property
    def model(self) -> str:
        return PROVIDERS[self.provider].model

    @property
    def terminal_marker(self) -> str | None:
        return PROTOCOLS[self.protocol].terminal_marker

    @property
    def stat_name(self) -> str:
        """Locust request name — unique per matrix point."""
        suffix = " [stream]" if self.stream else ""
        return f"{self.protocol}:{self.provider}{suffix}"


def _iter_combos() -> list[tuple[str, str, bool]]:
    """(protocol, provider, stream) points of the matrix."""
    combos: list[tuple[str, str, bool]] = []
    for protocol, provider in _EXPECTED_TIERS:
        supports_stream = PROTOCOLS[protocol].supports_stream
        combos.append((protocol, provider, False))
        if supports_stream:
            combos.append((protocol, provider, True))
    return combos


def _build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for protocol, provider, stream in _iter_combos():
        spec = PROVIDERS[provider]
        if protocol == "embeddings" and not spec.supports_embeddings:
            continue
        key = f"{protocol}_{provider}_{'stream' if stream else 'nonstream'}"
        tier = _EXPECTED_TIERS[(protocol, provider)]
        groups = [
            protocol,
            provider,
            "stream" if stream else "nonstream",
            tier,
        ]
        # ``native``/``wire_reuse``/``conversion`` coarse buckets: a compound
        # tier like ``wire_reuse+native_stream`` belongs to both buckets.
        for part in tier.split("+"):
            base = "native" if "native" in part else part
            if base not in groups:
                groups.append(base)
        if key in _CORE:
            groups.append("core")
        scenarios.append(
            Scenario(
                key=key,
                protocol=protocol,
                provider=provider,
                stream=stream,
                expected_tier=tier,
                groups=tuple(dict.fromkeys(groups)),
                description=f"{PROTOCOLS[protocol].path} -> {spec.name} ({tier})"
                + (" [stream]" if stream else ""),
            )
        )
    return scenarios


SCENARIOS: list[Scenario] = _build_scenarios()
SCENARIOS_BY_KEY: dict[str, Scenario] = {s.key: s for s in SCENARIOS}


def scenario_tokens() -> list[str]:
    """Every token ``select_scenarios`` accepts (groups + keys), sorted."""
    tokens: set[str] = {"all"}
    for s in SCENARIOS:
        tokens.add(s.key)
        tokens.update(s.groups)
    return sorted(tokens)


def select_scenarios(spec: str | None = None) -> list[Scenario]:
    """Resolve a comma-separated scenario spec into a list of scenarios.

    Tokens may be individual scenario keys (``chat_openai_compat_stream``) or
    groups (``messages``, ``native``, ``stream``, ``conversion``, ``core``,
    ``all``). Example: ``LOADTEST_SCENARIOS=messages,responses``.
    """
    if spec is None:
        spec = os.getenv("LOADTEST_SCENARIOS", "core")
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if not tokens or "all" in tokens:
        return list(SCENARIOS)

    selected: dict[str, Scenario] = {}
    unknown: list[str] = []
    for token in tokens:
        if token in SCENARIOS_BY_KEY:
            selected[token] = SCENARIOS_BY_KEY[token]
            continue
        matched = [s for s in SCENARIOS if token in s.groups]
        if matched:
            for s in matched:
                selected[s.key] = s
        else:
            unknown.append(token)
    if unknown:
        raise SystemExit(
            f"Unknown LOADTEST_SCENARIOS token(s): {', '.join(unknown)}.\n"
            f"Valid tokens: {', '.join(scenario_tokens())}"
        )
    return [SCENARIOS_BY_KEY[k] for k in sorted(selected)]


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def _build_large_content(target_tokens: int) -> str:
    # "lorem " is ~1 o200k token per word, so reps == target tokens. The fake
    # upstream's chars//4 counter reports ~1.5x that, which only feeds usage
    # records. Precomputed once at import: load generators should not spend
    # CPU building 300KB strings per request.
    return "lorem " * target_tokens


# Large-prompt mode (LiteLLM high-throughput benchmark shape): comma-separated
# token targets, e.g. LOADTEST_PROMPT_TOKENS="50000,75000,100000". 0 disables.
_LARGE_CONTENTS = [
    _build_large_content(int(t))
    for t in os.getenv("LOADTEST_PROMPT_TOKENS", "0").split(",")
    if int(t) > 0
]


def _request_content() -> str:
    if _LARGE_CONTENTS:
        return f"{uuid.uuid4()} {_LARGE_CONTENTS[len(_LARGE_CONTENTS) // 2]}"
    # UUID prefix defeats any response caching; long body exercises the
    # serialization hot path.
    return f"{uuid.uuid4()} This is a load test " * 30


def build_payload(scenario: Scenario) -> dict:
    """Build the client-wire request body for one scenario."""
    content = _request_content()
    model = scenario.model
    if scenario.protocol == "chat":
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        if scenario.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload
    if scenario.protocol == "messages":
        payload = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": content}],
        }
        if scenario.stream:
            payload["stream"] = True
        return payload
    if scenario.protocol == "responses":
        payload = {"model": model, "input": content}
        if scenario.stream:
            payload["stream"] = True
        return payload
    if scenario.protocol == "embeddings":
        return {"model": model, "input": content[:256]}
    raise ValueError(f"Unknown protocol: {scenario.protocol}")


def build_headers(scenario: Scenario, api_key: str) -> dict[str, str]:
    """Client headers for one scenario (auth + protocol-specific)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    headers.update(PROTOCOLS[scenario.protocol].headers)
    return headers


__all__ = [
    "PROTOCOLS",
    "PROVIDERS",
    "SCENARIOS",
    "SCENARIOS_BY_KEY",
    "ProtocolSpec",
    "ProviderSpec",
    "Scenario",
    "build_headers",
    "build_payload",
    "scenario_tokens",
    "select_scenarios",
]
