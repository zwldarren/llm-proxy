"""``prompt_cache_options`` only reaches upstreams that understand it (ADR-0066).

The field carries a cache lifetime (``ttl`` of ``30m``/``12h``) and a prewarm
flag. It is parsed into ``InternalRequest.extra`` because the internal model does
not hold it, so before this gate it was merged into every Chat Completions body —
including the strict third-party gateways that reject an unknown key with a 400.
The allow-list is the one ``prompt_cache_key`` already used.
"""

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.models.params import GenerationParams
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder

OPTIONS = {"mode": "explicit", "ttl": "12h"}


def _request(*, options: dict | None = OPTIONS) -> InternalRequest:
    extra = {"prompt_cache_options": options} if options is not None else {}
    return InternalRequest(
        model="gpt-6.2-sol",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
        extra=extra,
    )


def _body(base_url: str | None, *, options: dict | None = OPTIONS) -> dict:
    return OpenAIRequestBuilder().build(
        _request(options=options),
        BuildContext(
            model="gpt-6.2-sol",
            provider_name="openai",
            target_endpoint="chat_completions",
            base_url=base_url,
        ),
    )


def test_native_openai_receives_the_options() -> None:
    body = _body("https://api.openai.com/v1")
    assert body["prompt_cache_options"] == OPTIONS


def test_a_strict_gateway_does_not() -> None:
    body = _body("https://api.example-gateway.com/v1")
    assert "prompt_cache_options" not in body


def test_an_unknown_base_url_does_not() -> None:
    """No base URL means no known-compatible upstream, so the field is dropped."""
    assert "prompt_cache_options" not in _body(None)


def test_absent_options_leave_the_body_alone() -> None:
    assert "prompt_cache_options" not in _body("https://api.openai.com/v1", options=None)


def test_kimi_coding_path_accepts_it_like_prompt_cache_key() -> None:
    assert "prompt_cache_options" in _body("https://api.kimi.com/coding/v1")
