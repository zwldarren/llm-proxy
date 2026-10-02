"""Build the ``/v1``-only OpenAPI specification published in the docs site.

The FastAPI application is the source of truth for request schemas: every
model-validated endpoint contributes the exact Pydantic schema the server
validates against, so the documented request contract cannot drift from the
running contract. This module narrows the full application spec down to the
client-facing ``/v1`` surface, prunes unreachable components, and layers the
hand-authored fragments from :mod:`llm_proxy.api.openapi_overlays` on top.

Two guarantees are enforced here rather than documented by convention:

* path aliases (``/v1/v1/...``) and trailing-slash variants never reach the
  published spec — only the canonical path does;
* every canonical ``/v1`` operation must have an overlay, so adding a client
  endpoint without documenting it fails generation loudly instead of shipping
  an undocumented route.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_proxy.api.openapi_overlays import (
    COMPONENTS as OVERLAY_COMPONENTS,
)
from llm_proxy.api.openapi_overlays import (
    MULTIPART_REQUEST_BODIES,
    OPERATION_OVERLAYS,
    ROOT_SECURITY,
    SECURITY_SCHEMES,
    TAGS,
)
from llm_proxy.protocols.openai.schemas import (
    ImageEditRequestSchema,
    ImageGenerationRequestSchema,
)

#: Repository root, derived from this file's location (``src/llm_proxy/api/``).
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Where the generated spec is committed. VitePress imports it at build time.
DEFAULT_SPEC_PATH = REPO_ROOT / "docs" / "public" / "v1-openapi.json"

HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})

#: Endpoints registered with ``request_model=None`` whose body a Pydantic model
#: still validates inside ``parse_http_request``. Mapping them here keeps their
#: request schema generated from code instead of hand-written.
MODEL_BACKED_REQUEST_BODIES: dict[tuple[str, str], type[BaseModel]] = {
    ("/v1/images/generations", "post"): ImageGenerationRequestSchema,
    ("/v1/images/edits", "post"): ImageEditRequestSchema,
}

#: Error responses added to every operation when the endpoint did not declare
#: them. ``422`` is contributed by the generated spec for model-validated bodies.
_STANDARD_ERROR_STATUSES = ("400", "401", "429", "500")


def is_canonical_v1_path(path: str) -> bool:
    """Whether ``path`` is the canonical client-facing form of a ``/v1`` route.

    Excludes the alias forms the routers register for SDK compatibility
    (``/v1/v1/...``) and the trailing-slash variants that only exist to dodge
    the SPA catch-all.
    """
    return path.startswith("/v1/") and not path.endswith("/") and not path.startswith("/v1/v1/")


def canonical_operations(spec: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """The canonical ``/v1`` operations present in a generated spec, by path."""
    return {
        path: tuple(method for method in ops if method in HTTP_METHODS)
        for path, ops in spec.get("paths", {}).items()
        if is_canonical_v1_path(path)
    }


def _collect_refs(node: Any, into: set[str]) -> None:
    """Collect every ``#/components/schemas/<name>`` reference reachable from ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                prefix = "#/components/schemas/"
                if value.startswith(prefix):
                    into.add(value[len(prefix) :])
            else:
                _collect_refs(value, into)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, into)


def _error_envelope(path: str) -> dict[str, Any]:
    """The error component matching the protocol the path answers in."""
    name = "AnthropicErrorEnvelope" if path.startswith("/v1/messages") else "ErrorEnvelope"
    return {"$ref": f"#/components/schemas/{name}"}


def _build_request_body(
    path: str,
    method: str,
    extra_components: dict[str, Any],
) -> dict[str, Any] | None:
    """Assemble the request body for a canonical ``/v1`` operation, if any."""
    content: dict[str, Any] = {}

    model = MODEL_BACKED_REQUEST_BODIES.get((path, method))
    if model is not None:
        name = model.__name__
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        defs = schema.pop("$defs", {})
        extra_components[name] = schema
        extra_components.update(defs)
        content["application/json"] = {"schema": {"$ref": f"#/components/schemas/{name}"}}

    multipart = MULTIPART_REQUEST_BODIES.get((path, method))
    if multipart is not None:
        content.update(multipart["content"])

    if not content:
        return None
    return {"required": True, "content": content}


def _apply_overlay(
    op: dict[str, Any],
    path: str,
    method: str,
    overlay: dict[str, Any],
    extra_components: dict[str, Any],
) -> None:
    """Merge one hand-authored overlay into a generated operation."""
    if "summary" in overlay:
        op["summary"] = overlay["summary"]
    if "description" in overlay:
        op["description"] = overlay["description"]
    if "tags" in overlay:
        op["tags"] = overlay["tags"]

    request_body = _build_request_body(path, method, extra_components)
    if request_body is not None:
        op["requestBody"] = request_body

    generated_responses = op.get("responses", {})
    responses: dict[str, Any] = dict(overlay.get("responses", {}))

    # Streaming operations describe the same 200 with a second media type.
    sse_schema = overlay.get("sse")
    if sse_schema is not None and "200" in responses:
        responses["200"] = {
            **responses["200"],
            "content": {
                **responses["200"].get("content", {}),
                "text/event-stream": {"schema": sse_schema},
            },
        }

    # Preserve generated responses (notably the 422 validation envelope) that
    # the overlay did not override.
    for status, response in generated_responses.items():
        responses.setdefault(status, response)

    envelope = _error_envelope(path)
    for status in _STANDARD_ERROR_STATUSES:
        if status not in responses:
            responses[status] = {
                "description": "Error",
                "content": {"application/json": {"schema": envelope}},
            }

    op["responses"] = responses


def build_v1_spec(app: Any | None = None) -> dict[str, Any]:
    """Build the canonical ``/v1`` OpenAPI 3.1 specification.

    Args:
        app: A FastAPI application. Defaults to constructing the real app.

    Returns:
        The filtered, component-pruned, overlay-patched specification.

    Raises:
        RuntimeError: If a canonical ``/v1`` operation has no hand-authored
            overlay, which would otherwise publish an undocumented endpoint.
    """
    if app is None:
        from llm_proxy.api import create_app

        app = create_app()

    full = app.openapi()
    canonical = canonical_operations(full)

    undocumented = [
        f"{method.upper()} {path}"
        for path, methods in canonical.items()
        for method in methods
        if (path, method) not in OPERATION_OVERLAYS
    ]
    if undocumented:
        raise RuntimeError(
            "Canonical /v1 operations have no OpenAPI overlay in "
            "llm_proxy/api/openapi_overlays.py: " + ", ".join(sorted(undocumented))
        )

    extra_components: dict[str, Any] = dict(OVERLAY_COMPONENTS)
    paths: dict[str, Any] = {}
    for path, methods in canonical.items():
        ops: dict[str, Any] = {}
        for method in methods:
            op = dict(full["paths"][path][method])
            _apply_overlay(op, path, method, OPERATION_OVERLAYS[(path, method)], extra_components)
            ops[method] = op
        paths[path] = ops

    # Transitive closure of component references reachable from the retained
    # operations, so the published spec carries no admin-console schemas.
    reachable: set[str] = set()
    for ops in paths.values():
        _collect_refs(ops, reachable)

    schemas: dict[str, Any] = dict(extra_components)
    source_schemas = full.get("components", {}).get("schemas", {})
    pending = set(reachable)
    while pending:
        name = pending.pop()
        if name in schemas:
            continue
        definition = source_schemas.get(name)
        if definition is None:
            continue
        schemas[name] = definition
        nested: set[str] = set()
        _collect_refs(definition, nested)
        pending |= nested

    pruned = {name: schemas[name] for name in sorted(schemas)}

    return {
        "openapi": full.get("openapi", "3.1.0"),
        "info": {
            "title": "LLM Proxy API",
            "version": full["info"]["version"],
            "description": (
                "The client-facing contract of the proxy: the OpenAI Chat Completions, "
                "OpenAI Responses, and Anthropic Messages protocols plus the media "
                "endpoints, all served under `/v1`. Request schemas are generated from "
                "the same Pydantic models the server validates with; response schemas "
                "are descriptive and permissive, since upstream bodies are passed "
                "through. See https://zwldarren.github.io/llm-proxy/api/ for the full "
                "reference, including WebSocket and streaming behaviour."
            ),
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Local proxy (default)"},
        ],
        "tags": TAGS,
        "security": ROOT_SECURITY,
        "paths": paths,
        "components": {
            "securitySchemes": SECURITY_SCHEMES,
            "schemas": pruned,
        },
    }


def serialize_spec(spec: dict[str, Any]) -> str:
    """Render the spec as the exact text that is committed to the repository."""
    return json.dumps(spec, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Regenerate (or verify) the committed ``/v1`` OpenAPI document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SPEC_PATH),
        help="Destination path (defaults to docs/public/v1-openapi.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the destination is stale instead of writing it.",
    )
    args = parser.parse_args(argv)

    spec = build_v1_spec()
    text = serialize_spec(spec)
    output = Path(args.output)

    if args.check:
        current = output.read_text() if output.exists() else ""
        if current != text:
            print(
                f"{output} is out of date. Run `uv run llm-proxy-openapi` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"{output} is up to date.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(
        f"Wrote {output} ({len(spec['paths'])} paths, {len(spec['components']['schemas'])} schemas)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SPEC_PATH",
    "build_v1_spec",
    "canonical_operations",
    "is_canonical_v1_path",
    "main",
    "serialize_spec",
]
