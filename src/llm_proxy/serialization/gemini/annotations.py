"""Shared Gemini annotation extraction.

Converts Gemini ``citationMetadata`` / ``groundingMetadata`` into OpenAI-style
``url_citation`` annotations. Used by both the non-streaming response parser
and the streaming transformer so the two paths stay consistent.

Gemini measures citation offsets in **bytes** (CitationSource: "Index
indicates the start of the segment, measured in bytes"; Segment: "Start index
in the given Part, measured in bytes"). OpenAI ``url_citation`` offsets are
character offsets, so byte offsets are converted via UTF-8 decoding — without
this, citations misalign for any multi-byte text (e.g. Chinese).
"""

from typing import Any

from llm_proxy.models.types import UrlCitation


def _byte_offset_to_char(text: str, byte_offset: int) -> int:
    """Convert a UTF-8 byte offset into a character offset within *text*.

    Truncated multi-byte sequences at the slice boundary are ignored
    (``errors="ignore"``), matching how the API truncates segment output.
    """
    if byte_offset <= 0:
        return 0
    encoded = text.encode("utf-8")
    return len(encoded[:byte_offset].decode("utf-8", errors="ignore"))


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
            # startIndex/endIndex are byte offsets — convert to char offsets.
            start = _byte_offset_to_char(text, int(source.get("startIndex", 0) or 0))
            end_value = source.get("endIndex")
            end = _byte_offset_to_char(text, int(end_value)) if end_value is not None else end_index
            citation = UrlCitation(
                url=url,
                title=source.get("title", ""),
                start_index=start,
                end_index=end,
            )
            annotations.append({"type": "url_citation", "url_citation": citation.__dict__})

    grounding_metadata = candidate.get("groundingMetadata")
    if isinstance(grounding_metadata, dict):
        chunks = grounding_metadata.get("groundingChunks") or []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            # GroundingChunk variants per schema: web, retrievedContext, maps,
            # image (image search: sourceUri is the attribution page).
            source = (
                chunk.get("web")
                or chunk.get("retrievedContext")
                or chunk.get("maps")
                or chunk.get("image")
            )
            if not isinstance(source, dict):
                continue
            url = (
                source.get("uri")
                or source.get("url")
                or source.get("sourceUri")
                or source.get("imageUri")
            )
            if not url:
                continue
            citation = UrlCitation(
                url=url,
                title=source.get("title", ""),
                start_index=0,
                end_index=end_index,
            )
            annotations.append({"type": "url_citation", "url_citation": citation.__dict__})

    return annotations
