"""Upstream wire dialects of the OpenCode gateway, and how to pick one.

The OpenCode gateway (``https://opencode.ai``) fronts many upstream model
families and exposes each family on the wire its upstream speaks natively
(see the endpoint table in the Console models docs):

- Chat Completions (``/chat/completions``) — DeepSeek, GLM, Kimi, MiniMax,
  Mistral, the free/stealth models. This is also the one dialect every lane
  serves for every model, which makes it the universal fallback.
- OpenAI Responses (``/openai/v1/responses``) — GPT, Grok and Muse models.
- Anthropic Messages (``/anthropic/v1/messages``) — Claude and the Qwen
  Plus/Flash models.
- Google generateContent (``/google/v1beta/models/{model}``) — Gemini models.
- System One (``/systemone/v1/systemone``) — Jev evaluation models.

Which dialect a model speaks is per-model data, not per-family truth: the
same ``minimax-m2.7`` id is Chat Completions on the Console lane and
Anthropic Messages on the Go lane. Static prefix tables rot exactly there,
so the adapter resolves the dialect from data first (the models.dev catalog
the OpenCode team itself maintains — :mod:`llm_proxy.providers.opencode.catalog`)
and falls back to the prefix heuristics in this module only when the catalog
is unavailable or the model is unknown to it. Heuristics are compiled from
the catalog snapshot this module was written against and err toward Chat
Completions, the dialect the gateway normalizes everything to.
"""

import re
from enum import StrEnum

#: The v2 lane roots: the pay-as-you-go lane at ``/inference`` and the Go
#: subscription lane at ``/inference/go``. Each wire dialect hangs off its own
#: segment there (``/openai/v1``, ``/anthropic/v1``, ``/google/v1beta``,
#: ``/systemone/v1``) and the model list moves to ``<root>/v1/models``. Chat
#: Completions shares the OpenAI segment with Responses, so — unlike the
#: legacy ``/inference/v1`` root — it is not served at the base URL itself.
_V2_LANE_RE = re.compile(r"^(?P<root>https?://(?:[^/@]+\.)?opencode\.ai/inference(?:/go)?)/?$")

#: The legacy Console root prefixes the non-chat dialects into the path
#: (``/inference/<dialect>/<version>/...``) but keeps Chat Completions at the
#: base; the Zen (``/zen/v1``) and legacy Go (``/zen/go/v1``,
#: ``/inference/go/v1``) roots keep every dialect on one flat base. Only these
#: exact documented shapes trigger derivation; anything else (mirrors) derives
#: flat paths, and ``endpoint_base_urls`` overrides win over all of it.
_CONSOLE_LANE_RE = re.compile(r"^https?://(?:[^/@]+\.)?opencode\.ai/inference/v1/?$")


class Dialect(StrEnum):
    """The upstream wire dialect a model is served on."""

    CHAT = "chat"
    RESPONSES = "responses"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    SYSTEMONE = "systemone"


#: models.dev ``provider.npm`` values mapped to the dialect the OpenCode
#: gateway serves the model on. ``@ai-sdk/mistral`` maps to Chat Completions
#: because the gateway's own endpoint table lists Mistral models on the Chat
#: Completions endpoint. Anything unlisted degrades to Chat Completions.
NPM_DIALECTS: dict[str, Dialect] = {
    "@ai-sdk/openai": Dialect.RESPONSES,
    "@ai-sdk/anthropic": Dialect.ANTHROPIC,
    "@ai-sdk/google": Dialect.GOOGLE,
    "@ai-sdk/openai-compatible": Dialect.CHAT,
    "@ai-sdk/mistral": Dialect.CHAT,
}

#: Dialects that speak a chat-completion-shaped request the proxy can build.
#: The System One dialect is not one of them: a Jev model never takes a chat
#: body, and the adapter rejects such requests with a pointer at the System
#: One endpoint instead of shipping an upstream 400.
CHAT_CAPABLE: frozenset[Dialect] = frozenset(
    {Dialect.CHAT, Dialect.RESPONSES, Dialect.ANTHROPIC, Dialect.GOOGLE}
)

#: ``(pattern, dialect)`` rules consulted in order when the catalog has no
#: entry for a model. First match wins; the default is Chat Completions.
#: Compiled from the catalog snapshot below — the catalog overrides these at
#: runtime, so a new family that lands in models.dev needs no code change.
_ID_DIALECT_RULES: tuple[tuple[re.Pattern[str], Dialect], ...] = (
    (re.compile(r"^jev[-_.]"), Dialect.SYSTEMONE),
    (re.compile(r"^claude[-_.]"), Dialect.ANTHROPIC),
    (re.compile(r"^gemini[-_.]"), Dialect.GOOGLE),
    (re.compile(r"^(?:gpt[-_.]|grok[-_.](?!code)|muse[-_.]spark)"), Dialect.RESPONSES),
)

#: Dialect assumed when nothing else knows: the one every lane serves for
#: every model.
DEFAULT_DIALECT = Dialect.CHAT

#: models.dev provider entries keyed by lane. ``opencode`` is the Console
#: pay-as-you-go lane, ``opencode-go`` the subscription lane.
CATALOG_ENTRY_CONSOLE = "opencode"
CATALOG_ENTRY_GO = "opencode-go"


def infer_dialect_from_id(model: str | None) -> Dialect:
    """Guess the dialect from the model id alone (no catalog, no network)."""
    if not model:
        return DEFAULT_DIALECT
    lowered = model.lower()
    for pattern, dialect in _ID_DIALECT_RULES:
        if pattern.match(lowered):
            return dialect
    return DEFAULT_DIALECT


def dialect_from_npm(npm: str | None) -> Dialect:
    """Map a models.dev ``provider.npm`` marker onto a gateway dialect."""
    if not npm:
        return DEFAULT_DIALECT
    return NPM_DIALECTS.get(npm, DEFAULT_DIALECT)


def _normalize_base(base_url: str | None) -> str:
    """A configured base URL without surrounding space or a trailing slash."""
    return (base_url or "").strip().rstrip("/")


def _v2_root(base: str) -> str | None:
    """The v2 lane root ``base`` names, or None when it names a legacy one."""
    match = _V2_LANE_RE.match(base)
    return match.group("root") if match else None


def derive_dialect_roots(base_url: str | None) -> dict[Dialect, str]:
    """Derive the per-dialect endpoint roots from the configured base URL.

    On the v2 lane roots (``https://opencode.ai/inference``,
    ``https://opencode.ai/inference/go``) every dialect hangs off its own
    segment, Chat Completions included
    (``/openai/v1/chat/completions``). On the legacy Console root
    (``https://opencode.ai/inference/v1``) the non-chat dialects hang off
    their own segments while Chat Completions rides the base URL; on every
    other lane (Zen ``/zen/v1``, legacy Go ``/zen/go/v1``) they all share one
    flat root (``/messages``, ``/responses``, ...).
    """
    base = _normalize_base(base_url)
    if root := _v2_root(base):
        return {
            Dialect.CHAT: f"{root}/openai/v1",
            Dialect.RESPONSES: f"{root}/openai/v1",
            Dialect.ANTHROPIC: f"{root}/anthropic/v1",
            Dialect.GOOGLE: f"{root}/google/v1beta",
            Dialect.SYSTEMONE: f"{root}/systemone/v1",
        }
    if _CONSOLE_LANE_RE.match(base):
        root = base[: -len("/v1")]
        return {
            Dialect.CHAT: base,
            Dialect.RESPONSES: f"{root}/openai/v1",
            Dialect.ANTHROPIC: f"{root}/anthropic/v1",
            Dialect.GOOGLE: f"{root}/google/v1beta",
            Dialect.SYSTEMONE: f"{root}/systemone/v1",
        }
    return {dialect: base for dialect in Dialect}


def derive_models_url(base_url: str | None) -> str:
    """The gateway's model-list URL for one configured base URL.

    The v2 lanes version it under the root (``/inference/go/v1/models``);
    every other root serves it flat (``/zen/go/v1/models``).
    """
    base = _normalize_base(base_url)
    if root := _v2_root(base):
        return f"{root}/v1/models"
    return f"{base}/models"


__all__ = [
    "CATALOG_ENTRY_CONSOLE",
    "CATALOG_ENTRY_GO",
    "CHAT_CAPABLE",
    "DEFAULT_DIALECT",
    "Dialect",
    "NPM_DIALECTS",
    "dialect_from_npm",
    "derive_dialect_roots",
    "derive_models_url",
    "infer_dialect_from_id",
]
