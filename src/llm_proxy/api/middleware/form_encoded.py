"""Middleware converting x-www-form-urlencoded bodies to JSON for /v1/responses.

The OpenResponses reference allows request parameters to be supplied as either
``application/json`` or ``application/x-www-form-urlencoded``. FastAPI's
pydantic body parsing only handles JSON, so this middleware transparently
converts form-encoded POST bodies on /v1/responses into JSON before the route
handler runs. Fields that carry structured values (input, tools, tool_choice,
...) are JSON-decoded when the form value is valid JSON.

Note: the converted body is written back through the ``BodyReader``, so the
replacement is what the downstream pydantic parser reads.
"""

import json
from urllib.parse import parse_qsl

from starlette.requests import Request
from starlette.responses import JSONResponse

from llm_proxy.api.middleware.asgi_utils import (
    BodyReader,
    CoreASGIMiddleware,
    adapt_http_middleware,
)

_FORM_ENCODED = "application/x-www-form-urlencoded"

# Form fields whose values are structured (JSON) rather than plain strings.
_JSON_FIELDS = frozenset(
    {
        "input",
        "tools",
        "tool_choice",
        "metadata",
        "text",
        "reasoning",
        "stream_options",
        "include",
    }
)


def _coerce_form_value(key: str, value: str) -> object:
    if key in _JSON_FIELDS:
        try:
            return json.loads(value)
        except ValueError, TypeError:
            return value
    return value


async def _dispatch(request: Request, body: BodyReader) -> JSONResponse | None:
    """Convert form-encoded /v1/responses POST bodies to JSON."""
    content_type = request.headers.get("content-type", "")
    if not (
        request.method == "POST"
        and request.url.path == "/v1/responses"
        and content_type.startswith(_FORM_ENCODED)
    ):
        return None

    try:
        raw = await body.read()
        form = {k: _coerce_form_value(k, v) for k, v in parse_qsl(raw.decode("utf-8"))}
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid form-encoded request body.",
                    "type": "invalid_request",
                }
            },
        )

    # Replace the body and content-type in place so the downstream pydantic
    # parser sees a JSON body.
    body.replace(json.dumps(form).encode("utf-8"))
    request.scope["headers"] = [
        (b"content-type", b"application/json"),
        *[h for h in request.scope.get("headers", []) if h[0].lower() != b"content-type"],
    ]
    return None


class FormEncodedMiddleware(CoreASGIMiddleware):
    """Pure-ASGI form-encoded to JSON conversion for /v1/responses."""

    async def dispatch(self, request: Request, body: BodyReader) -> JSONResponse | None:
        return await _dispatch(request, body)


#: ``(request, call_next)`` adapter kept for existing call sites and tests.
form_encoded_middleware = adapt_http_middleware(_dispatch)
