"""Tests for shared block conversion (``_shared_conversion.try_convert_block``).

Focused on replayed server-tool results: a client echoing a prior turn's
``web_search_tool_result`` verbatim must reach non-Anthropic providers with its
snippet intact, not as a per-item title placeholder, or the model loses the
facts it already fetched and searches again.
"""

import orjson

from llm_proxy.models import ToolResultBlock
from llm_proxy.models.content_blocks.anthropic_builtin import (
    WebSearchResultContentBlock,
    WebSearchToolResultBlock,
)
from llm_proxy.serialization._shared_conversion import try_convert_block


def _result_block(encrypted: str) -> WebSearchToolResultBlock:
    return WebSearchToolResultBlock(
        tool_use_id="ws_1",
        content=[
            WebSearchResultContentBlock(
                url="https://reuters.com/world/iran/",
                title="Iran War",
                encrypted_content=encrypted,
            )
        ],
    )


def test_replayed_web_search_result_decodes_the_snippet() -> None:
    # "relevant snippet" base64-encoded, as the interceptor stores it.
    converted = try_convert_block(_result_block("cmVsZXZhbnQgc25pcHBldA=="))

    assert isinstance(converted, ToolResultBlock)
    assert converted.tool_use_id == "ws_1"
    assert converted.is_error is False
    # The exact shape the proxy's own continuation sends upstream.
    assert orjson.loads(converted.content) == {
        "results": [
            {
                "url": "https://reuters.com/world/iran/",
                "title": "Iran War",
                "snippet": "relevant snippet",
            }
        ]
    }


def test_replayed_web_search_result_without_decodable_snippet_keeps_title_and_url() -> None:
    # Real Anthropic ``encrypted_content`` is opaque, not the proxy's base64
    # placeholder: the result must still be delivered, snippet empty.
    converted = try_convert_block(_result_block("!!!not-base64!!!"))

    assert isinstance(converted, ToolResultBlock)
    payload = orjson.loads(converted.content)
    assert payload["results"][0]["title"] == "Iran War"
    assert payload["results"][0]["url"] == "https://reuters.com/world/iran/"
    assert payload["results"][0]["snippet"] == ""


def test_web_search_error_payload_is_kept_verbatim() -> None:
    converted = try_convert_block(
        WebSearchToolResultBlock(
            tool_use_id="ws_1",
            content='{"type": "web_search_tool_result_error", "error_code": "unavailable"}',
            is_error=True,
        )
    )

    assert isinstance(converted, ToolResultBlock)
    assert converted.is_error is True
    assert "web_search_tool_result_error" in converted.content
