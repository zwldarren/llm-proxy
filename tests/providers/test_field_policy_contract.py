"""Parametrized contract test: every endpoint method honors field policy.

Calls the REAL endpoint method (not _build_outbound_body directly) so that
if an endpoint is ever refactored to bypass the dispatch chokepoint, the
corresponding parametrization FAILS.  Each transport is mocked at the
HTTP layer appropriate for the endpoint type.

Endpoints covered:
  - chat (chat_completion)           → mock _post_json_with_retry
  - embedding (embeddings)           → mock _post_json_with_retry
  - speech (speech)                  → mock httpx client + retry
  - transcription (transcription)    → mock httpx client + retry
  - translation (translation)        → mock httpx client + retry
  - image_generation (image_generation) → mock _post_json_with_retry
  - image_edit (image_edit)          → mock _post_json_with_retry

Also covers the Gemini adapter (different transport pattern: direct
_get_client + _with_retry) for its image_generation and image_edit
endpoints, which were historically bypassing the chokepoint.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
    InternalTranslationRequest,
    Message,
    TextBlock,
)
from llm_proxy.models.image import ImageEditSource
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

# ── Sentinel ────────────────────────────────────────────────────────────

SENTINEL_KEY = "__synthetic_sentinel__"
SENTINEL_VAL = "contract-test-v7"


# ── Adapter factory ─────────────────────────────────────────────────────


def _adapter(policy: str) -> OpenAICompatibleBase:
    return OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        unknown_fields_policy=policy,
    )


# ── Request factories (each returns a request with the sentinel extra) ──


def _chat_req() -> InternalRequest:
    return InternalRequest(
        model="gpt-4o",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _embedding_req() -> InternalEmbeddingRequest:
    return InternalEmbeddingRequest(
        model="text-embedding-3-small",
        input="hi",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _speech_req() -> InternalSpeechRequest:
    return InternalSpeechRequest(
        model="tts-1",
        input="Hello",
        voice="alloy",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _transcription_req() -> InternalTranscriptionRequest:
    return InternalTranscriptionRequest(
        model="whisper-1",
        file=b"\x00\x00",
        filename="sentinel.mp3",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _translation_req() -> InternalTranslationRequest:
    return InternalTranslationRequest(
        model="whisper-1",
        file=b"\x00\x00",
        filename="sentinel.mp3",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _image_gen_req() -> InternalImageRequest:
    return InternalImageRequest(
        model="dall-e-3",
        prompt="a cat",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _image_edit_req() -> InternalImageEditRequest:
    return InternalImageEditRequest(
        model="dall-e-2",
        prompt="a cat",
        images=[ImageEditSource(file_id="img_123")],
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


# ── Mock response factories ─────────────────────────────────────────────


def _chat_response() -> dict:
    return {
        "id": "chatcmpl-contract-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _embedding_response() -> dict:
    return {
        "data": [{"embedding": [0.0], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _image_response() -> dict:
    return {"data": [], "created": 1}


# ── Response lookup ─────────────────────────────────────────────────────

_RESPONSES: dict[str, dict] = {
    "chat": _chat_response(),
    "embedding": _embedding_response(),
    "image_generation": _image_response(),
    "image_edit": _image_response(),
    # speech/transcription/translation handle responses via mock client
}


# ── Mock helpers ────────────────────────────────────────────────────────


def _mock_json_transport(adapter: OpenAICompatibleBase, monkeypatch, name: str, capture: dict):
    """Mock JSON transport helpers to capture the posted body.

    Covers both ``_post_json_with_retry`` (image endpoints) and
    ``_post_json_response_with_retry`` (chat/embedding endpoints, which return
    the raw response so headers can be captured).
    """

    async def fake_post(_url, _headers, body):
        capture["body"] = body
        return _RESPONSES[name]

    async def fake_post_response(_url, _headers, body):
        capture["body"] = body
        resp = MagicMock()
        resp.json.return_value = _RESPONSES[name]
        resp.headers = {}
        return resp

    monkeypatch.setattr(adapter, "_post_json_with_retry", fake_post)
    monkeypatch.setattr(adapter, "_post_json_response_with_retry", fake_post_response)


def _mock_client_transport(adapter: OpenAICompatibleBase, monkeypatch, capture: dict, **kw):
    """Mock _get_client, _check_response_status, and _retry.execute for client-based endpoints.

    Returns the mock client so callers can inspect post args.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = kw.get("json_data", {"text": "ok"})
    resp.content = kw.get("content", b"fake-bytes")
    resp.headers = MagicMock()
    resp.headers.get.return_value = kw.get("content_type", "application/json")

    client = MagicMock()
    client.post = AsyncMock(return_value=resp)

    async def fake_get_client():
        return client

    async def fake_check_status(_resp):
        pass

    monkeypatch.setattr(adapter, "_get_client", fake_get_client)
    monkeypatch.setattr(adapter, "_check_response_status", fake_check_status)
    monkeypatch.setattr(adapter._retry, "execute", lambda op, *a, **kw: op())

    capture["client"] = client
    return client


# ── Endpoint metadata ───────────────────────────────────────────────────
# (name, factory, method_name, transport_type)
#   transport_type: "json" → mock _post_json_with_retry
#                   "client" → mock httpx client + retry

JSON_ENDPOINTS = [
    ("chat", _chat_req, "chat_completion", "json"),
    ("embedding", _embedding_req, "embeddings", "json"),
    ("image_generation", _image_gen_req, "image_generation", "json"),
    ("image_edit", _image_edit_req, "image_edit", "json"),
]

CLIENT_ENDPOINTS = [
    ("speech", _speech_req, "speech", "client"),
    ("transcription", _transcription_req, "transcription", "client"),
    ("translation", _translation_req, "translation", "client"),
]

ALL_ENDPOINTS = JSON_ENDPOINTS + CLIENT_ENDPOINTS


# ═══════════════════════════════════════════════════════════════════════
# error policy — raises BEFORE HTTP (no transport mocking needed)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name,mock_type",
    ALL_ENDPOINTS,
    ids=[e[0] for e in ALL_ENDPOINTS],
)
async def test_error_policy_rejects_unknown_field(name, factory, method_name, mock_type):
    """Every endpoint raises ProviderError for unknown fields under 'error' policy.

    The error is raised in _build_outbound_body (inside the endpoint method)
    before any HTTP call — so no transport mock is needed.
    """
    a = _adapter("error")
    req = factory()
    method = getattr(a, method_name)

    with pytest.raises(ProviderError, match="unknown request fields"):
        await method(req)


# ═══════════════════════════════════════════════════════════════════════
# passthrough policy — sentinel MUST be in the posted body/data
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name,mock_type",
    ALL_ENDPOINTS,
    ids=[e[0] for e in ALL_ENDPOINTS],
)
async def test_passthrough_keeps_sentinel(name, factory, method_name, mock_type, monkeypatch):
    """Every endpoint passes the sentinel extra field through under 'passthrough' policy."""
    a = _adapter("passthrough")
    req = factory()
    method = getattr(a, method_name)
    capture: dict = {}

    if mock_type == "json":
        _mock_json_transport(a, monkeypatch, name, capture)
        await method(req)
        body = capture["body"]
        assert body.get(SENTINEL_KEY) == SENTINEL_VAL, (
            f"[{name}] sentinel missing from posted JSON body"
        )

    elif mock_type == "client":
        client = _mock_client_transport(a, monkeypatch, capture)
        await method(req)
        posted_data = client.post.call_args.kwargs.get("data") or client.post.call_args.kwargs.get(
            "json"
        )
        assert posted_data is not None, f"[{name}] no data/json in client.post call"
        assert posted_data.get(SENTINEL_KEY) == SENTINEL_VAL, (
            f"[{name}] sentinel missing from posted data"
        )


# ═══════════════════════════════════════════════════════════════════════
# ignore policy — sentinel MUST NOT be in the posted body/data
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name,mock_type",
    ALL_ENDPOINTS,
    ids=[e[0] for e in ALL_ENDPOINTS],
)
async def test_ignore_strips_sentinel(name, factory, method_name, mock_type, monkeypatch):
    """Every endpoint strips the sentinel extra field under 'ignore' policy."""
    a = _adapter("ignore")
    req = factory()
    method = getattr(a, method_name)
    capture: dict = {}

    if mock_type == "json":
        _mock_json_transport(a, monkeypatch, name, capture)
        await method(req)
        body = capture["body"]
        assert SENTINEL_KEY not in body, f"[{name}] sentinel leaked through under ignore policy"

    elif mock_type == "client":
        client = _mock_client_transport(a, monkeypatch, capture)
        await method(req)
        posted_data = client.post.call_args.kwargs.get("data") or client.post.call_args.kwargs.get(
            "json"
        )
        assert posted_data is not None, f"[{name}] no data/json in client.post call"
        assert SENTINEL_KEY not in posted_data, (
            f"[{name}] sentinel leaked through under ignore policy"
        )


def _contains_key(obj: Any, target_key: str) -> bool:
    """Recursively search dicts and lists for a key at any nesting level."""
    if isinstance(obj, dict):
        if target_key in obj:
            return True
        for v in obj.values():
            if _contains_key(v, target_key):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _contains_key(item, target_key):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Gemini adapter — uses _get_client + _with_retry transport pattern
# ═══════════════════════════════════════════════════════════════════════

from llm_proxy.providers.gemini.adapter import GeminiAdapter  # noqa: E402


def _gemini_adapter(policy: str) -> GeminiAdapter:
    return GeminiAdapter(
        api_key="test-key",
        unknown_fields_policy=policy,
    )


def _gemini_image_gen_req() -> InternalImageRequest:
    return InternalImageRequest(
        model="gemini-2.0-flash-exp-image",
        prompt="a cat",
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _gemini_image_edit_req() -> InternalImageEditRequest:
    return InternalImageEditRequest(
        model="gemini-2.0-flash-exp-image",
        prompt="edit this cat",
        images=[ImageEditSource(file_id="img_123")],
        extra={SENTINEL_KEY: SENTINEL_VAL},
    )


def _gemini_image_response() -> dict:
    """Valid Gemini generateContent response with an image."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mime_type": "image/png",
                                "data": "ZmFrZV9iYXNlNjRfZGF0YQ==",
                            }
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


def _mock_gemini_transport(adapter: GeminiAdapter, monkeypatch, capture: dict):
    """Mock Gemini's _get_client + _with_retry + _check_response_status pattern.

    Returns the mock client so callers can inspect post kwargs.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _gemini_image_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=resp)

    async def fake_get_client():
        return mock_client

    async def fake_check_status(_resp):
        pass

    monkeypatch.setattr(adapter, "_get_client", fake_get_client)
    monkeypatch.setattr(adapter, "_check_response_status", fake_check_status)
    monkeypatch.setattr(adapter._retry, "execute", lambda op, *a, **kw: op())

    capture["client"] = mock_client
    return mock_client


GEMINI_IMAGE_ENDPOINTS = [
    ("gemini-image_generation", _gemini_image_gen_req, "image_generation"),
    ("gemini-image_edit", _gemini_image_edit_req, "image_edit"),
]


# ── error policy (no transport needed) ──────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name",
    GEMINI_IMAGE_ENDPOINTS,
    ids=[e[0] for e in GEMINI_IMAGE_ENDPOINTS],
)
async def test_gemini_error_policy_rejects_unknown_field(name, factory, method_name):
    """Gemini image endpoints raise ProviderError under 'error' policy."""
    a = _gemini_adapter("error")
    req = factory()
    method = getattr(a, method_name)

    with pytest.raises(ProviderError, match="unknown request fields"):
        await method(req)


# ── passthrough policy — sentinel MUST be in posted body ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name",
    GEMINI_IMAGE_ENDPOINTS,
    ids=[e[0] for e in GEMINI_IMAGE_ENDPOINTS],
)
async def test_gemini_passthrough_keeps_sentinel(name, factory, method_name, monkeypatch):
    """Gemini image endpoints pass sentinel through under 'passthrough' policy."""
    a = _gemini_adapter("passthrough")
    req = factory()
    method = getattr(a, method_name)
    capture: dict = {}

    mock_client = _mock_gemini_transport(a, monkeypatch, capture)
    await method(req)

    posted_body = mock_client.post.call_args.kwargs["json"]
    assert posted_body.get(SENTINEL_KEY) == SENTINEL_VAL, (
        f"[{name}] sentinel missing from posted JSON body"
    )


# ── ignore policy — sentinel MUST NOT be in posted body ─────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,factory,method_name",
    GEMINI_IMAGE_ENDPOINTS,
    ids=[e[0] for e in GEMINI_IMAGE_ENDPOINTS],
)
async def test_gemini_ignore_strips_sentinel(name, factory, method_name, monkeypatch):
    """Gemini image endpoints strip sentinel under 'ignore' policy."""
    a = _gemini_adapter("ignore")
    req = factory()
    method = getattr(a, method_name)
    capture: dict = {}

    mock_client = _mock_gemini_transport(a, monkeypatch, capture)
    await method(req)

    posted_body = mock_client.post.call_args.kwargs["json"]
    assert not _contains_key(posted_body, SENTINEL_KEY), (
        f"[{name}] sentinel leaked through under ignore policy (recursive check)"
    )
