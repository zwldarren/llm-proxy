"""Shared OpenAI Responses-API response-level field names.

The OpenAI provider serializer captures these fields from the upstream
response (they are not carried by the internal model), and the OpenResponses
protocol serializer echoes them back onto the emitted ResponseResource. Both
sides must list the same keys or a field silently drops on the converted path;
they agree on this one constant.

All nine are Response-object fields in the official OpenAI API spec
(``moderation``, ``prompt_cache_diagnostics``, ``access_programs``,
``prompt_cache_options``, ``prompt_cache_retention``, ``conversation``,
``image``, ``tool_usage``, ``user``). ``background_options`` is deliberately
absent: it is a request-only field that the response object does not echo.
"""

RESPONSE_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "moderation",
    "prompt_cache_diagnostics",
    "access_programs",
    "prompt_cache_options",
    "prompt_cache_retention",
    "conversation",
    "image",
    "tool_usage",
    "user",
)

__all__ = ["RESPONSE_PASSTHROUGH_KEYS"]
