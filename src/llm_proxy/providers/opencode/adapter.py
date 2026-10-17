"""OpenCode gateway adapter — one upstream, one wire dialect per model.

The OpenCode gateway fronts model families whose upstreams speak different
wire protocols: Claude (and the Qwen Plus/Flash models) on Anthropic
Messages, GPT/Grok/Muse on OpenAI Responses, Gemini on generateContent, Jev
on System One, and everything else on Chat Completions — which is also the
one dialect every lane serves for every model. The dialect a request takes
is therefore a property of the *model*, resolved per request from, in
order:

1. a learned correction (a dialect the gateway rejected before, see below),
2. the models.dev catalog (``provider.npm`` per model — the OpenCode team's
   own data, so new families route correctly with no code change),
3. the model-id prefix heuristics in :mod:`.dialect` (Chat Completions by
   default — the universal dialect).

A dialect the gateway rejects with a model-flavored 400/404 is corrected at
runtime: the request is rebuilt for Chat Completions and re-sent once, and
the correction is remembered for the process lifetime, so an unknown or
reclassified model self-heals instead of failing.

Lanes. The Console lane (default base URL ``https://opencode.ai/inference/v1``)
prefixes dialects into the path (``/inference/anthropic/v1/messages``); the
Zen (``/zen/v1``) and Go (``/zen/go/v1``, ``/inference/go/v1``) lanes keep
every dialect on one flat root. :func:`.dialect.derive_dialect_roots`
derives the right shape, and every dialect's URL is overridable through the
provider's ``endpoint_base_urls`` (keys ``chat_completion``, ``responses``,
``anthropic_messages``, ``google_generate``, ``systemone``).

Headers. Chat/Responses/System One authenticate with ``Authorization:
Bearer``; the gateway's Anthropic route reads ``x-api-key`` and its Google
route ``x-goog-api-key``, so those dialects send the Bearer header *and*
their dialect's auth header with the same key. The edge also checks the
OpenCode CLI's client identity (``User-Agent: opencode/...``,
``x-opencode-client``, ``x-opencode-project``, per-request
``x-opencode-session``/``x-opencode-request``) — free-tier models are
rejected with ``403 FreeTierError`` without it (:mod:`.identity`). All of
these are defaults: ``custom_headers`` wins over any of them.
"""

import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from orjson import JSONDecodeError

from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.models import ConversionTier, InternalRequest, InternalResponse
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import extract_rate_limit_headers
from llm_proxy.providers.capabilities import SystemOneCapabilityMixin
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase
from llm_proxy.providers.opencode import catalog
from llm_proxy.providers.opencode.dialect import (
    CATALOG_ENTRY_CONSOLE,
    CHAT_CAPABLE,
    DEFAULT_DIALECT,
    Dialect,
    derive_dialect_roots,
    infer_dialect_from_id,
)
from llm_proxy.providers.opencode.identity import identity_headers
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers import get_provider_serializer
from llm_proxy.streaming.sse_parse import iter_sse_data_events

logger = get_logger(__name__)

# The shared Chat Completions serializer registers under the "openrouter"/
# "deepseek" names from serialization/providers/chat_completions.py — a module
# the serializer registry's on-demand fallback (serialization/<name>/serializer.py)
# never reaches, so it is imported explicitly before the module-level lookups
# below (providers/__init__ does the same for its own import order). Not
# circular: it imports only from serialization/*.
import llm_proxy.serialization.providers.chat_completions  # noqa: F401, E402

# Dialect serializers, resolved once at import (the providers package imports
# serializer modules before adapter modules; direct imports hit the registry's
# on-demand fallback).
_CHAT_SERIALIZER = get_provider_serializer("openrouter")
_RESPONSES_SERIALIZER = get_provider_serializer("openai")
_ANTHROPIC_SERIALIZER = get_provider_serializer("anthropic")
_GOOGLE_SERIALIZER = get_provider_serializer("gemini")
_DIALECT_SERIALIZERS: dict[Dialect, Any] = {
    Dialect.CHAT: _CHAT_SERIALIZER,
    Dialect.RESPONSES: _RESPONSES_SERIALIZER,
    Dialect.ANTHROPIC: _ANTHROPIC_SERIALIZER,
    Dialect.GOOGLE: _GOOGLE_SERIALIZER,
}

#: Client protocol -> the only dialect that protocol may be forwarded to
#: verbatim (the native passthrough tiers).
_PROTOCOL_DIALECT: dict[str, Dialect] = {
    "anthropic": Dialect.ANTHROPIC,
    "openresponses": Dialect.RESPONSES,
}

#: Endpoint keys of the provider's ``endpoint_base_urls`` overrides, per dialect.
_DIALECT_OVERRIDE_KEYS: dict[Dialect, str] = {
    Dialect.RESPONSES: "responses",
    Dialect.ANTHROPIC: "anthropic_messages",
    Dialect.GOOGLE: "google_generate",
    Dialect.SYSTEMONE: "systemone",
}

#: Message fragments (lowercased) that identify a "this model is not served
#: on this endpoint" rejection — the trigger for the dialect correction.
_MISMATCH_MARKERS = (
    "not found",
    "no such",
    "unknown model",
    "invalid model",
    "does not exist",
    "not supported",
    "unsupported",
    "is not available",
    "not available for",
)

# Learned dialect corrections, process-wide: (base_url, model) -> dialect.
# Bounded FIFO — the gateway's catalog is a few hundred models, so the bound
# is never hit in practice; it exists so a hostile model stream cannot grow
# the table unboundedly.
_CORRECTIONS: OrderedDict[tuple[str, str], Dialect] = OrderedDict()
_CORRECTIONS_MAX = 512
_CORRECTIONS_LOCK = threading.Lock()


def _remember_correction(base_url: str, model: str, dialect: Dialect) -> None:
    key = (base_url or "", model)
    with _CORRECTIONS_LOCK:
        _CORRECTIONS.pop(key, None)
        _CORRECTIONS[key] = dialect
        while len(_CORRECTIONS) > _CORRECTIONS_MAX:
            _CORRECTIONS.popitem(last=False)


def _corrected_dialect(base_url: str, model: str | None) -> Dialect | None:
    if not model:
        return None
    with _CORRECTIONS_LOCK:
        return _CORRECTIONS.get((base_url or "", model))


def _fallback_dialects(origin: Dialect) -> tuple[Dialect, ...]:
    """Dialects to try after ``origin`` is rejected, most likely first.

    Chat Completions comes first: it is the one dialect every lane serves for
    every model. The remaining chat-capable dialects follow, so a model that
    lives only on a native dialect (an Anthropic-only Qwen, say) is still
    found when the catalog is unreachable and the heuristics guessed wrong.
    """
    if origin is Dialect.CHAT:
        return (Dialect.ANTHROPIC, Dialect.RESPONSES, Dialect.GOOGLE)
    chat_capable = (Dialect.CHAT, Dialect.ANTHROPIC, Dialect.RESPONSES, Dialect.GOOGLE)
    return tuple(d for d in chat_capable if d is not origin)


def _is_dialect_mismatch(error: ProviderError) -> bool:
    """Whether an upstream rejection means "this model is not on this endpoint".

    Deliberately narrow: a 404 always matches (a route the lane does not
    implement — the Zen proxy rewrites its upstream 404s into 400s with the
    body preserved, which the markers catch), while a 400 triggers the
    correction only when its body is model-flavored, so ordinary invalid
    requests and auth failures surface as themselves instead of silently
    re-routing.
    """
    if error.status_code == 404:
        return True
    if error.status_code != 400:
        return False
    message = (error.message or "").lower()
    if "model" not in message:
        return False
    return any(marker in message for marker in _MISMATCH_MARKERS)


def _raise_systemone_chat_error(model: str | None) -> None:
    """Reject a chat-shaped request for a System One model.

    Jev models evaluate a ``state`` against typed questions instead of
    generating text; no chat body can reach them on any lane. Fail with the
    endpoint that can serve them instead of a gateway 400.
    """
    raise ValidationError(
        message=(
            f"Model '{model}' is a System One evaluation model; it is only "
            "served on the gateway's /v1/systemone endpoint and cannot take "
            "chat requests."
        ),
        code="invalid_request_error",
        status_code=400,
    )


class OpenCodeGatewayAdapter(SystemOneCapabilityMixin, NativePassthroughChatBase):
    """OpenCode gateway adapter with per-model dialect inference.

    Unregistered shared base: the registered types are the lanes
    (``opencode-zen``, ``opencode-go``), each declaring its default base URL
    and the models.dev catalog entry its models are published under. Chat
    Completions is the base dialect (shared with every
    ``NativePassthroughChatBase`` provider); Anthropic Messages, OpenAI
    Responses, Google generateContent and System One are dispatched per
    model. The dialect is resolved once per request (``_resolve_dialect``,
    async — it may fetch the catalog) and then read synchronously everywhere
    else (``_active_dialect``); adapter instances are created per request, so
    the attribute is request-scoped by construction.
    """

    _DEFAULT_PROVIDER_NAME = ""

    #: Which models.dev provider entry this lane's models are published under
    #: (``opencode`` — the Console models the Zen lane serves — or
    #: ``opencode-go``). Set per registered type; the catalog entry decides
    #: where dialects and pricing are read from.
    CATALOG_ENTRY = CATALOG_ENTRY_CONSOLE

    #: Anthropic Messages and OpenAI Responses are served natively — but only
    #: for models whose dialect matches the client protocol; the model gate
    #: lives in ``supports_native_request`` / ``native_streaming_veto``.
    native_protocols = frozenset({"anthropic", "openresponses"})

    #: The dialect resolved for the request in flight. Adapter instances are
    #: created per request (``adapter_factory``), so this is request-scoped.
    _active_dialect: Dialect | None = None

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        base = self._base_url or self.DEFAULT_BASE_URL
        self._dialect_roots = derive_dialect_roots(base)
        self._catalog_entry = self.CATALOG_ENTRY
        self._catalog_tables: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Model catalog
    # ------------------------------------------------------------------

    async def list_models(self, client: Any = None) -> list[Any]:
        """List the gateway's models, enriched from the models.dev catalog.

        The gateway's own ``/v1/models`` answers with bare ids; the catalog
        contributes the display names, context windows and per-1M pricing the
        admin UI renders. Enrichment is best-effort: a catalog failure leaves
        the plain gateway listing in place.
        """
        models = await super().list_models(client)
        url = self._catalog_url
        if not url:
            return models
        try:
            http = client if client is not None else await self._get_client()
            tables = await catalog.load_catalog(http, url=url)
        except Exception as exc:  # pragma: no cover - defensive; load_catalog guards itself
            logger.debug(f"{self.provider_name}: model catalog enrichment skipped: {exc}")
            return models
        entry = tables.get(self._catalog_entry, {})
        for model in models:
            info = entry.get(model.id)
            if info is None:
                continue
            if info.name:
                model.name = info.name
            if info.description:
                model.description = info.description
            if info.context_length:
                model.context_length = info.context_length
            if info.pricing is not None:
                model.pricing = info.pricing
        return models

    # ------------------------------------------------------------------
    # Dialect resolution
    # ------------------------------------------------------------------

    @property
    def _catalog_url(self) -> str:
        """The models.dev catalog location; empty disables the catalog."""
        value = self._extra_config.get("opencode_catalog_url")
        if value is None:
            return catalog.DEFAULT_CATALOG_URL
        return str(value)

    def _dialect_for_request(self, request: InternalRequest | None) -> Dialect:
        """Synchronous dialect resolution: corrections, warm catalog, heuristics.

        Runs inside sync hooks the pipeline reaches before the adapter's async
        entry points (the streaming plan's veto) — it never fetches, only
        reads what is already cached process-wide.
        """
        model = request.model if request is not None else None
        corrected = _corrected_dialect(self._base_url or "", model)
        if corrected is not None:
            return corrected
        tables = self._catalog_tables or catalog.peek()
        if tables:
            entry = catalog.lookup(tables, self._catalog_entry, model)
            if entry is not None:
                return entry.dialect
        return infer_dialect_from_id(model)

    async def _resolve_dialect(self, request: InternalRequest) -> Dialect:
        """Async dialect resolution — the per-request entry point.

        Warms the catalog (single-flight, timeout-guarded; failures degrade
        to heuristics) before the model is looked up, so the first request of
        a process resolves against live data rather than the compiled
        snapshot.
        """
        model = request.model
        corrected = _corrected_dialect(self._base_url or "", model)
        if corrected is not None:
            return corrected
        url = self._catalog_url
        if url:
            client = await self._get_client()
            self._catalog_tables = await catalog.load_catalog(client, url=url)
            entry = catalog.lookup(self._catalog_tables, self._catalog_entry, model)
            if entry is not None:
                return entry.dialect
        return infer_dialect_from_id(model)

    def _current_dialect(self, request: InternalRequest | None) -> Dialect:
        """The dialect for sync body/header building this request."""
        return self._active_dialect or self._dialect_for_request(request)

    def _remember_chat_correction(self, model: str | None, dialect: Dialect) -> None:
        """Record the dialect a model was actually served on after a fallback.

        The gateway's own catalog data overrides this on the next TTL window;
        until then the correction spares every later request the failed
        attempts. Only fallback successes are recorded, so an auth failure or
        a genuine bad request can never poison the table.
        """
        if model:
            _remember_correction(self._base_url or "", model, dialect)

    # ------------------------------------------------------------------
    # Dialect endpoints
    # ------------------------------------------------------------------

    def _dialect_url(self, dialect: Dialect, model: str | None, *, stream: bool) -> str:
        """The upstream URL for one dialect."""
        override_key = _DIALECT_OVERRIDE_KEYS.get(dialect)
        if override_key and override_key in self._endpoint_base_urls:
            url = self._resolve_endpoint_url(override_key, "", model=model)
            if dialect is Dialect.GOOGLE and url.endswith(":generateContent") and stream:
                # Streaming rides the same route with a different method
                # suffix; the override names the non-streaming URL.
                url = f"{url[: -len(':generateContent')]}:streamGenerateContent?alt=sse"
            return url
        root = self._dialect_roots[dialect]
        match dialect:
            case Dialect.RESPONSES:
                return f"{root}/responses"
            case Dialect.ANTHROPIC:
                return f"{root}/messages"
            case Dialect.GOOGLE:
                method = "streamGenerateContent?alt=sse" if stream else "generateContent"
                return f"{root}/models/{model}:{method}"
            case Dialect.SYSTEMONE:
                return f"{root}/systemone"
        return self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT, model=model)

    def _responses_url(self, model: str | None = None) -> str:
        return self._dialect_url(Dialect.RESPONSES, model, stream=False)

    def _anthropic_messages_url(self, model: str | None = None) -> str:
        return self._dialect_url(Dialect.ANTHROPIC, model, stream=False)

    def _systemone_url(self, request: Any) -> str:
        return self._dialect_url(Dialect.SYSTEMONE, request.model, stream=False)

    # ------------------------------------------------------------------
    # Upstream headers
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        """Dialect-aware headers: auth shape, API version, client identity.

        The Anthropic route reads ``x-api-key`` and the Google route
        ``x-goog-api-key``; both dialects also carry the Bearer header with
        the same key, because the lanes disagree on which one they read.
        The OpenCode CLI identity headers go out on every request (the free
        tier rejects requests without them); the operator's
        ``custom_headers`` wins over any identity default.
        """
        dialect = self._active_dialect or DEFAULT_DIALECT
        match dialect:
            case Dialect.ANTHROPIC:
                headers = super()._build_headers("x-api-key", "")
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
            case Dialect.GOOGLE:
                headers = super()._build_headers("x-goog-api-key", "")
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"
            case _:
                headers = super()._build_headers(auth_header, auth_prefix)

        if dialect is Dialect.ANTHROPIC:
            # The API version describes the request: a client that pinned its
            # own (Claude Code does) keeps it; otherwise the current stamp.
            existing = {key.lower() for key in headers}
            if "anthropic-version" not in existing:
                headers["anthropic-version"] = "2023-06-01"

        custom_lower = {key.lower() for key in (self._custom_headers or {})}
        for key, value in identity_headers().items():
            if key.lower() not in custom_lower:
                headers[key] = value
        return headers

    # ------------------------------------------------------------------
    # Body building
    # ------------------------------------------------------------------

    def _get_serializer(self):
        """The default (Chat Completions) serializer for non-chat capabilities."""
        return _CHAT_SERIALIZER

    def _build_chat_context(self, request: InternalRequest) -> BuildContext:
        """Dialect-aware build context: the serializer, wire compatibility and
        target endpoint follow the model's dialect, not the adapter's."""
        dialect = self._current_dialect(request)
        if dialect not in CHAT_CAPABLE:
            _raise_systemone_chat_error(request.model)
        serializer = _DIALECT_SERIALIZERS[dialect]
        return BuildContext.from_request(
            request,
            base_url=self._base_url,
            provider_name=self.provider_name,
            provider_type=self._adapter_type,
            # The Responses dialect builds Responses bodies; the Chat
            # Completions value is otherwise the (ignored) default for the
            # Anthropic/Google builders.
            target_endpoint="responses" if dialect is Dialect.RESPONSES else "chat_completions",
            unknown_fields_policy=self._resolve_field_policy(),
            unsupported_block_policy=self._resolve_block_policy(),
            supported_content_blocks=serializer.supported_content_blocks,
            compatible_protocols=serializer.compatible_protocols,
            response_passthrough=bool(self._extra_config.get("response_passthrough", True)),
            reasoning_object=self.REASONING_OBJECT,
        )

    def _build_chat_raw(self, request: InternalRequest, context: BuildContext) -> dict[str, Any]:
        """Full conversion through the model's dialect serializer."""
        dialect = self._current_dialect(request)
        if dialect is Dialect.CHAT:
            return super()._build_chat_raw(request, context)
        return _DIALECT_SERIALIZERS[dialect].build_provider_request(request, context)

    def _build_dialect_body(self, request: InternalRequest) -> dict[str, Any]:
        """The outbound body for a converted non-chat dialect request.

        Built for the request's active dialect (``self._active_dialect``),
        which every entry point sets before building.
        """
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("Expected JSON body for chat request")
        return outbound.json_body

    def _stream_body(self, request: InternalRequest) -> dict[str, Any]:
        """Streaming body, plan-aware and dialect-aware.

        The Chat Completions pipeline (reasoning normalization, echo,
        include_usage) applies to the chat dialect only; the other dialects
        build through their serializer. Native-tier requests arrive prepared
        by the passthrough seam.
        """
        dialect = self._current_dialect(request)
        if dialect is Dialect.CHAT:
            return super()._stream_body(request)
        plan = plan_conversion(self, request, context=self._build_chat_context(request))
        if plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH:
            outbound = self._build_outbound_body(request, request_type="chat")
            if outbound.json_body is None:
                raise ValueError("Expected JSON body for chat request")
            return outbound.json_body
        body = self._build_dialect_body(request)
        body["stream"] = True
        return body

    # ------------------------------------------------------------------
    # Native passthrough gating (dialect must match the client protocol)
    # ------------------------------------------------------------------

    def supports_native_request(
        self, protocol_name: str | None, request: InternalRequest | None = None
    ) -> bool:
        """Native verbatim forwarding, gated on the model's dialect.

        An Anthropic client may ride the native Messages tier only for
        Anthropic-dialect models, an OpenResponses client only for
        Responses-dialect models; anything else is translated instead.
        """
        if not super().supports_native_request(protocol_name, request):
            return False
        if request is None:
            return True
        return self._dialect_for_request(request) is _PROTOCOL_DIALECT.get(protocol_name or "")

    def native_streaming_veto(self, request: InternalRequest) -> bool:
        """Veto native streaming when the model's dialect mismatches the protocol."""
        dialect = self._dialect_for_request(request)
        expected = _PROTOCOL_DIALECT.get(request.protocol_name or "")
        if expected is not None and dialect is not expected:
            return True
        if request.protocol_name == "openai" and dialect is not Dialect.CHAT:
            return True
        return super().native_streaming_veto(request)

    def _native_request_parts(
        self, request: InternalRequest, *, stream: bool
    ) -> tuple[str, dict[str, Any]]:
        """Native endpoint parts, stamping the dialect for header building."""
        self._active_dialect = _PROTOCOL_DIALECT.get(request.protocol_name or "")
        return super()._native_request_parts(request, stream=stream)

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Native SSE passthrough with a plan-aware body.

        Unlike the generic base — which only ever forwards the stashed raw
        body — the rebuilt tiers (a conversation materialized from the
        response store) stream the freshly built dialect body instead, so a
        proxy-local id on the request never reaches the upstream.
        """
        if request.protocol_name == "openai":
            return await super().stream_chat_completion_native(
                request, cancel_token=cancel_token, **kwargs
            )
        dialect = _PROTOCOL_DIALECT.get(request.protocol_name or "")
        if dialect is None:
            raise NotImplementedError(
                f"{self.provider_name} does not support native streaming "
                f"for '{request.protocol_name}'"
            )
        self._active_dialect = dialect
        url = self._dialect_url(dialect, request.model, stream=False)
        body = self._stream_body(request)
        body["stream"] = True
        return self._with_retry_generator(
            lambda: self._stream_raw_sse(url, body, cancel_token),
            cancel_token=cancel_token,
        )

    # ------------------------------------------------------------------
    # Non-streaming chat completion
    # ------------------------------------------------------------------

    async def chat_completion(self, request: InternalRequest, **kwargs: Any) -> InternalResponse:
        dialect = await self._resolve_dialect(request)
        self._active_dialect = dialect
        if dialect not in CHAT_CAPABLE:
            _raise_systemone_chat_error(request.model)
        if dialect is not Dialect.CHAT:
            # Native verbatim forwarding for a matching client protocol (an
            # Anthropic client against a Claude model, an OpenResponses client
            # against a GPT model); everything else is translated. The chat
            # dialect has no native request tier — its wire-reuse tiers live
            # inside the chain's chat attempt.
            plan = plan_conversion(self, request, context=self._build_chat_context(request))
            if plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH:
                return await self._native_completion(request)
        return await self._complete_with_dialect(request, dialect)

    async def _complete_with_dialect(
        self, request: InternalRequest, dialect: Dialect
    ) -> InternalResponse:
        """Converted completion on a non-chat dialect, with dialect-chain fallback.

        The inferred dialect is tried first; a model-flavored rejection walks
        the fallback chain (Chat Completions, then the remaining dialects).
        The first error is re-raised when every attempt is rejected — it is
        the truthful diagnostic; the later ones are this adapter's speculations.
        """
        first_error: ProviderError | None = None
        for attempt in (dialect, *_fallback_dialects(dialect)):
            self._active_dialect = attempt
            try:
                if attempt is Dialect.CHAT:
                    # The full Chat Completions path — wire-reuse tiers and
                    # reasoning normalization included — not a bare rebuild.
                    result = await super().chat_completion(request)
                    self._learn_served_dialect(request, dialect, attempt)
                    return result
                result = await self._post_dialect_attempt(request, attempt)
            except ProviderError as exc:
                if not _is_dialect_mismatch(exc):
                    raise
                if first_error is None:
                    first_error = exc
                continue
            self._learn_served_dialect(request, dialect, attempt)
            return result
        assert first_error is not None
        raise first_error

    async def _post_dialect_attempt(
        self, request: InternalRequest, dialect: Dialect
    ) -> InternalResponse:
        """Build, post and parse one converted non-chat dialect attempt."""
        serializer = _DIALECT_SERIALIZERS[dialect]
        url = self._dialect_url(dialect, request.model, stream=False)
        body = self._build_dialect_body(request)
        response = await self._post_json_response_with_retry(url, self._build_headers(), body)
        data = response.json()
        result = self._parse_response(
            serializer,
            data,
            model=request.model,
            request_id=request.request_id,
            request=request,
        )
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    # ------------------------------------------------------------------
    # Converted streaming per dialect
    # ------------------------------------------------------------------

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        dialect = await self._resolve_dialect(request)
        self._active_dialect = dialect
        if dialect not in CHAT_CAPABLE:
            _raise_systemone_chat_error(request.model)
        return self._with_retry_generator(
            lambda: self._stream_dialect(request, dialect, cancel_token, context, kwargs),
            cancel_token=cancel_token,
        )

    async def _stream_dialect(
        self,
        request: InternalRequest,
        dialect: Dialect,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Converted streaming, with the dialect-chain fallback.

        Every origin walks the same chain (its own dialect first, then the
        rest), so a correction that lands on Chat Completions still finds a
        native-dialect-only model when the gateway rejects it there. The
        chain fires only before the first chunk is yielded — an upstream
        rejection surfaces in the stream status check, so a fallback never
        interleaves with bytes already delivered. After the first chunk the
        stream belongs to the client and every error propagates.
        """
        kwargs = kwargs or {}
        first_error: ProviderError | None = None
        for attempt in (dialect, *_fallback_dialects(dialect)):
            self._active_dialect = attempt
            chunks_yielded = False
            try:
                if attempt is Dialect.CHAT:
                    async for chunk in self._chat_fallback_stream(
                        request, cancel_token, context, kwargs
                    ):
                        chunks_yielded = True
                        yield chunk
                    self._learn_served_dialect(request, dialect, attempt)
                    return
                async for chunk in self._stream_dialect_attempt(request, attempt, cancel_token):
                    chunks_yielded = True
                    yield chunk
                self._learn_served_dialect(request, dialect, attempt)
                return
            except ProviderError as exc:
                if chunks_yielded or not _is_dialect_mismatch(exc):
                    raise
                if first_error is None:
                    first_error = exc
                continue
        assert first_error is not None
        raise first_error

    def _learn_served_dialect(
        self, request: InternalRequest, inferred: Dialect, served: Dialect
    ) -> None:
        """Record a fallback success so later requests skip the failed attempts."""
        if served is inferred:
            return
        logger.info(
            f"{self.provider_name}: model '{request.model}' is served on the "
            f"{served.value} endpoint, not {inferred.value}; correction learned"
        )
        self._remember_chat_correction(request.model, served)

    async def _stream_dialect_attempt(
        self,
        request: InternalRequest,
        dialect: Dialect,
        cancel_token=None,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """One converted streaming attempt on a non-chat dialect."""
        serializer = _DIALECT_SERIALIZERS[dialect]
        url = self._dialect_url(dialect, request.model, stream=True)
        body = self._stream_body(request)
        converter = serializer.get_chunk_converter(
            model=request.model, request_id=request.request_id or ""
        )
        async for frame in self._stream_raw_sse(url, body, cancel_token):
            for chunk in self._convert_dialect_frame(frame, dialect, converter):
                yield chunk
        for chunk in converter.finalize_chunks():
            yield chunk
        yield "[DONE]"

    async def _chat_fallback_stream(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """The Chat Completions streaming path — the chain's terminal attempt."""
        stream = await super().stream_chat_completion(
            request, cancel_token=cancel_token, context=context, **(kwargs or {})
        )
        async for chunk in stream:
            yield chunk

    @staticmethod
    def _convert_dialect_frame(
        frame: str, dialect: Dialect, converter: Any
    ) -> list[dict[str, Any]]:
        """Turn one upstream SSE block into canonical chunks via the dialect converter.

        All three wire shapes share the SSE ``event:``/``data:`` framing, so
        one parser covers them; the converters key on the payload's own shape.
        The Responses converter additionally reads the frame's event type (the
        OpenAI adapter feeds it the same way).
        """
        chunks: list[dict[str, Any]] = []
        try:
            for event_type, payload in iter_sse_data_events(frame):
                if not isinstance(payload, dict):
                    continue
                if event_type or "type" in payload:
                    payload["event_type"] = event_type or payload.get("type")
                chunk = converter.convert_chunk(payload)
                if chunk is not None:
                    chunks.append(chunk)
        except JSONDecodeError:
            logger.debug(f"{dialect.value} SSE frame failed to parse, skipped: {frame[:200]}")
            return []
        return chunks


__all__ = ["OpenCodeGatewayAdapter"]
