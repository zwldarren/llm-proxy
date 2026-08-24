"""Shared base for the GLM platform adapters (z.ai international + 智谱 bigmodel.cn).

Both sites expose the same endpoint layout (docs.z.ai, docs.bigmodel.cn):

- Chat Completions: ``/api/paas/v4/chat/completions`` (pay-as-you-go) or
  ``/api/coding/paas/v4/chat/completions`` (GLM Coding Plan subscription)
- Anthropic Messages: ``/api/anthropic/v1/messages``
- OpenAI Responses: ``/api/v1/responses`` (Coding Plan only)

Auth quirk: the Anthropic-compatible endpoints are documented with the
Anthropic-style ``x-api-key`` header (智谱 curl examples) while the
OpenAI-compatible endpoints use ``Authorization: Bearer`` — and z.ai's Claude
Code setup uses ``ANTHROPIC_AUTH_TOKEN`` (Bearer). Sending both headers to
every endpoint is harmless (each side reads the one it supports), so
``_build_headers`` adds ``x-api-key`` alongside the default Bearer header.
This single override also covers the native streaming path, which builds its
headers via ``_build_headers`` inside ``_stream_raw_sse``.
"""

from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


class GLMBase(NativePassthroughChatBase):
    """GLM platform base: shared header quirk; concrete adapters add URLs."""

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        headers = super()._build_headers(auth_header, auth_prefix)
        if self._api_key:
            headers.setdefault("x-api-key", self._api_key)
        return headers


__all__ = ["GLMBase"]
