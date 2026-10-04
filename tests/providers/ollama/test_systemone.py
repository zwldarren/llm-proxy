"""Tests for Ollama's System One support (``/v1/systemone``, Ollama v0.35+).

Ollama serves the shared TypeSafe/OpenRouter wire format, but from a different
path (the OpenAI-compatibility prefix, not the native ``/api`` one) and with two
optional fields of its own (``images``, ``keep_alive``). These tests pin both
halves of that: the Ollama fields survive the outbound field policy, and the
OpenRouter-only ones do not.
"""

from typing import Any

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.providers.ollama.adapter import OllamaAdapter

QUESTION = {
    "type": "choice",
    "instructions": "Which label fits this ticket?",
    "criteria": {"billing": "Payments and refunds", "bug": "Software errors"},
}
STATE = "Our checkout has returned 500 errors since 9am."


def _request(**extra: Any) -> InternalSystemOneRequest:
    return InternalSystemOneRequest(
        model="nimble",
        state=STATE,
        questions={"label": QUESTION},
        extra=dict(extra),
    )


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter()


class TestOllamaSystemOneEndpoint:
    def test_default_base_url_and_path(self, adapter):
        assert adapter._systemone_url(_request()) == "http://localhost:11434/v1/systemone"

    def test_follows_configured_base_url(self):
        # Operators point the proxy at a remote daemon; the path must not pick
        # up the native /api prefix that chat and embeddings use.
        adapter = OllamaAdapter(base_url="http://gpu-box:11434")
        assert adapter._systemone_url(_request()) == "http://gpu-box:11434/v1/systemone"


class TestOllamaSystemOneBody:
    def test_core_fields_and_ollama_extras(self, adapter):
        outbound = adapter._build_outbound_body(
            _request(images=["aGVsbG8="], keep_alive="5m"), request_type="systemone"
        )
        assert outbound.json_body == {
            "model": "nimble",
            "state": STATE,
            "questions": {"label": QUESTION},
            "images": ["aGVsbG8="],
            "keep_alive": "5m",
        }

    def test_core_fields_only_when_no_extras(self, adapter):
        outbound = adapter._build_outbound_body(_request(), request_type="systemone")
        assert outbound.json_body == {
            "model": "nimble",
            "state": STATE,
            "questions": {"label": QUESTION},
        }

    def test_strips_openrouter_only_fields(self, adapter):
        outbound = adapter._build_outbound_body(
            _request(
                provider={"order": ["TypeSafe"]},
                session_id="s1",
                trace={"trace_id": "t1"},
                user="u1",
            ),
            request_type="systemone",
        )
        assert outbound.json_body == {
            "model": "nimble",
            "state": STATE,
            "questions": {"label": QUESTION},
        }

    def test_chat_extras_are_not_exempt_for_systemone(self, adapter):
        # The embedding exemptions (truncate/options) belong to /api/embed; a
        # chat leftover on a System One request is still stripped.
        outbound = adapter._build_outbound_body(
            _request(options={"num_ctx": 4096}), request_type="systemone"
        )
        assert "options" not in outbound.json_body


class TestOllamaSystemOneCall:
    async def test_posts_and_parses(self, adapter, mock_response_cls, monkeypatch):
        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            response = mock_response_cls(
                json_data={
                    "model": "nimble",
                    "answers": {
                        "label": {
                            "type": "choice",
                            "choice": "bug",
                            "probabilities": {"billing": 0.0125, "bug": 0.9781},
                            "confidence": 0.8906,
                        }
                    },
                    "usage": {"input_tokens": 174, "output_tokens": 1},
                }
            )
            response.headers = {"x-ratelimit-remaining": "10"}
            return response

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.systemone(_request(keep_alive="5m"))

        assert captured["url"] == "http://localhost:11434/v1/systemone"
        assert captured["json"]["keep_alive"] == "5m"
        assert result.model == "nimble"
        assert result.answers["label"]["choice"] == "bug"
        assert result.usage is not None
        assert (result.usage.input_tokens, result.usage.output_tokens) == (174, 1)
        # Ollama reports no generation id/provider and no cost, so none is
        # invented for it.
        assert result.id is None
        assert result.provider is None
        assert "openrouter_cost" not in result.provider_info
        assert result.provider_info["_rate_limit_headers"] == {"x-ratelimit-remaining": "10"}

    async def test_upstream_error_message_propagates(self, adapter, mock_response_cls, monkeypatch):
        # Ollama reports errors as ``{"error": "<string>"}``, unlike the nested
        # OpenAI shape; the shared translator must surface the message as-is.
        async def mock_post(url, **kwargs):
            return mock_response_cls(
                status_code=413,
                text_data='{"error": "request body must not exceed 64 KiB without images"}',
            )

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        with pytest.raises(ProviderError) as excinfo:
            await adapter.systemone(_request())
        assert excinfo.value.status_code == 413
        assert "64 KiB" in excinfo.value.message
