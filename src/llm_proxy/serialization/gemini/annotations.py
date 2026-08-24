"""Shared Gemini annotation extraction.

Converts Gemini ``citationMetadata`` / ``groundingMetadata`` into OpenAI-style
``url_citation`` annotations. Used by both the non-streaming response parser
and the streaming transformer so the two paths stay consistent.
"""

from typing import Any

from llm_proxy.models.types import UrlCitation


def extract_gemini_annotations(candidate: dict[str, Any], text: str = "") -> list[dict[str, Any]]:
    """Convert Gemini citationMetadata/groundingMetadata to OpenAI-style annotations."""
    annotations: list[dict[str, Any]] = []
    end_index = len(text)

    citation_metadata = candidate.get("citationMetadata")
    if isinstance(citation_metadata, dict):
        for source in citation_metadata.get("citationSources", []) or []:
            if not isinstance(source, dict):
                continue
            url = source.get("uri") or source.get("url")
            if not url:
                continue
            citation = UrlCitation(
                url=url,
                title=source.get("title", ""),
                start_index=source.get("startIndex", 0),
                end_index=source.get("endIndex", end_index),
            )
            annotations.append({"type": "url_citation", "url_citation": citation.__dict__})

    grounding_metadata = candidate.get("groundingMetadata")
    if isinstance(grounding_metadata, dict):
        chunks = grounding_metadata.get("groundingChunks") or []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web") or chunk.get("retrievedContext")
            if not isinstance(web, dict):
                continue
            url = web.get("uri") or web.get("url")
            if not url:
                continue
            citation = UrlCitation(
                url=url,
                title=web.get("title", ""),
                start_index=0,
                end_index=end_index,
            )
            annotations.append({"type": "url_citation", "url_citation": citation.__dict__})

    return annotations
