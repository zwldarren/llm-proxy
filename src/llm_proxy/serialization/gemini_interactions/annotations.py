"""Shared inline-annotation conversion for the Gemini Interactions API.

Converts the Interactions API's inline content ``annotations`` (``url_citation``
entries referenced from ``model_output`` content items and
``text_annotation_delta`` stream events) into OpenAI-style url_citation dicts.
Used by both the response parser and the streaming converter so the two paths
stay consistent.
"""

from typing import Any

from llm_proxy.models.types import UrlCitation


def content_annotations_to_openai(
    annotations: list[Any], end_index: int
) -> list[dict[str, Any]] | None:
    """Convert inline url_citation annotations to OpenAI-style dicts.

    Returns None when no annotation converts (nothing to attach).
    """
    converted: list[dict[str, Any]] = []
    for anno in annotations:
        if not isinstance(anno, dict) or anno.get("type") != "url_citation":
            continue
        # The migration guide emits "uri" while the API reference says
        # "url"; accept both (mirrors the legacy annotation extractor).
        url = anno.get("url") or anno.get("uri")
        if not url:
            continue
        citation = UrlCitation(
            url=url,
            title=anno.get("title", ""),
            start_index=anno.get("start_index", 0),
            end_index=anno.get("end_index", end_index),
        )
        converted.append({"type": "url_citation", "url_citation": citation.__dict__})
    return converted or None
