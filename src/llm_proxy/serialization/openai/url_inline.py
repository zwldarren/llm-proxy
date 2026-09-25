"""Download-and-inline for Chat Completions document/file URL sources.

Some Chat Completions upstreams accept only inline base64 on their ``file``
part — OpenAI and the generic ``openai-compatible`` adapter — so an external URL
cannot be forwarded as-is. The provider body builder is synchronous, so this
async pass runs in the adapter *before* the body is built and rewrites the
internal request's URL sources into base64 in place.

Only providers flagged by :func:`chat_document_url_needs_download` are touched;
providers with a native URL field keep the URL and providers with no document
part at all cannot be helped by inlining. Downloads are SSRF-protected inside
:func:`~llm_proxy.http.client.download_image_as_base64`.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from llm_proxy.core.utils import as_http_url
from llm_proxy.http.client import download_image_as_base64
from llm_proxy.models import DocumentBlock, FileBlock, InternalRequest
from llm_proxy.models.types import DocumentSource
from llm_proxy.observability.logger import get_logger
from llm_proxy.serialization.openai.converter import (
    chat_document_url_needs_download,
    chat_file_media_type_accepted,
)

logger = get_logger(__name__)


def _collect_url_sources(
    request: InternalRequest,
) -> list[tuple[int, int, str, str]]:
    """Return ``(msg_idx, block_idx, kind, url)`` for each URL source to inline."""
    targets: list[tuple[int, int, str, str]] = []
    for msg_idx, message in enumerate(request.conversation.messages):
        for block_idx, block in enumerate(message.content):
            if isinstance(block, DocumentBlock) and block.source.type == "url":
                url = as_http_url(block.source.data)
                if url:
                    targets.append((msg_idx, block_idx, "document", url))
            elif isinstance(block, FileBlock):
                url = as_http_url(block.file_url) or as_http_url(block.file_data)
                if url:
                    targets.append((msg_idx, block_idx, "file", url))
    return targets


def _base64_payload(data_url: str) -> str:
    return data_url.split(",", 1)[1] if "," in data_url else data_url


async def inline_chat_document_urls(
    request: InternalRequest,
    *,
    provider_type: str,
    client: Any,
) -> int:
    """Inline URL document/file sources for *provider_type* in place.

    Returns the number of sources replaced. A source is left untouched when its
    host has no document part, its file part takes a URL natively, the download
    fails, or the upstream does not accept the downloaded media type — the
    converter then degrades that block as before.
    """
    if not chat_document_url_needs_download(provider_type):
        return 0
    targets = _collect_url_sources(request)
    if not targets:
        return 0

    unique_urls = list(dict.fromkeys(url for _, _, _, url in targets))
    results = await asyncio.gather(
        *(download_image_as_base64(client, url) for url in unique_urls),
        return_exceptions=True,
    )
    downloaded: dict[str, tuple[str, str]] = {}
    for url, result in zip(unique_urls, results, strict=True):
        if isinstance(result, BaseException):
            logger.debug(f"Could not download document URL {url}: {result}")
            continue
        if result is not None:
            downloaded[url] = result

    inlined = 0
    for msg_idx, block_idx, kind, url in targets:
        result = downloaded.get(url)
        if result is None:
            continue
        data_url, media_type = result
        if not chat_file_media_type_accepted(provider_type, media_type):
            # e.g. a non-PDF document for OpenAI: inlining would not help, so let
            # the converter degrade it instead of emitting a part the API rejects.
            continue
        message = request.conversation.messages[msg_idx]
        block = message.content[block_idx]
        if kind == "document":
            new_source = DocumentSource(
                type="base64",
                data=_base64_payload(data_url),
                media_type=media_type,
            )
            message.content[block_idx] = dataclasses.replace(block, source=new_source)
        else:
            message.content[block_idx] = dataclasses.replace(
                block, file_data=data_url, file_url=None
            )
        inlined += 1

    if inlined:
        logger.debug(f"Inlined {inlined} document URL source(s) for {provider_type}")
    return inlined


__all__ = ["inline_chat_document_urls"]
