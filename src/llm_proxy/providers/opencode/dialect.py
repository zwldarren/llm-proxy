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

# The Console lane prefixes the dialect into the path
# (``/inference/<dialect>/<version>/...``); the Zen and Go lanes keep every
# dialect on one flat root (``/zen/v1/messages``, ``/zen/go/v1/responses``).
# Only this exact documented lane shape triggers prefix derivation; anything
# else (mirrors, /zen/v1, /inference/go/v1) derives flat paths, and
# ``endpoint_base_urls`` overrides win over both.
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


def derive_dialect_roots(base_url: str | None) -> dict[Dialect, str]:
    """Derive the per-dialect endpoint roots from the configured base URL.

    Chat always rides the base URL itself. On the documented Console lane
    (``https://opencode.ai/inference/v1``) the other dialects hang off their
    own segment roots (``/inference/anthropic/v1/messages``, ...); on every
    other lane (Zen ``/zen/v1``, Go ``/zen/go/v1``, ``/inference/go/v1``)
    they hang off the same flat root (``/messages``, ``/responses``, ...).
    """
    base = (base_url or "").strip().rstrip("/")
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


__all__ = [
    "CATALOG_ENTRY_CONSOLE",
    "CATALOG_ENTRY_GO",
    "CHAT_CAPABLE",
    "DEFAULT_DIALECT",
    "Dialect",
    "NPM_DIALECTS",
    "dialect_from_npm",
    "derive_dialect_roots",
    "infer_dialect_from_id",
]
