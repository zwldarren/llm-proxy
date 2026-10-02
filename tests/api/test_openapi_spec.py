"""Drift guards for the published ``/v1`` OpenAPI document.

The document at ``docs/public/v1-openapi.json`` is generated from the running
FastAPI app, so these tests are what keep it honest: the published path set must
match the routes the app actually registers, alias/admin routes must never leak
in, and the committed artifact must equal a fresh regeneration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytest

from llm_proxy.api.openapi_docs import (
    DEFAULT_SPEC_PATH,
    build_v1_spec,
    canonical_operations,
    is_canonical_v1_path,
    serialize_spec,
)


@lru_cache(maxsize=1)
def _full_spec() -> dict[str, Any]:
    """The unfiltered FastAPI spec — every route the app serves."""
    from llm_proxy.api import create_app

    return create_app().openapi()


@lru_cache(maxsize=1)
def _v1_spec() -> dict[str, Any]:
    """The published, ``/v1``-only spec."""
    return build_v1_spec()


class _StubApp:
    """Minimal stand-in for a FastAPI app exposing only ``openapi()``."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self._spec = spec

    def openapi(self) -> dict[str, Any]:
        return self._spec


def test_published_paths_are_exactly_the_canonical_v1_routes():
    expected = canonical_operations(_full_spec())
    published = {path: tuple(ops) for path, ops in _v1_spec()["paths"].items()}

    assert set(published) == set(expected)
    for path, methods in published.items():
        assert set(methods) == set(expected[path]), f"method set drifted for {path}"


def test_alias_routes_exist_but_are_filtered_out():
    full_paths = set(_full_spec()["paths"])

    # Sanity check: the filter has real work to do, these aliases do exist.
    assert "/v1/v1/chat/completions" in full_paths
    assert "/v1/chat/completions/" in full_paths

    published = _v1_spec()["paths"]
    assert "/v1/v1/chat/completions" not in published
    assert "/v1/chat/completions/" not in published


def test_no_non_client_routes_leak_into_the_spec():
    for path in _v1_spec()["paths"]:
        assert is_canonical_v1_path(path), path
        assert not path.startswith("/api"), path
        assert not path.startswith("/v1/v1"), path
        assert not path.endswith("/"), path


def test_every_operation_documents_a_success_response():
    for path, ops in _v1_spec()["paths"].items():
        for method, op in ops.items():
            where = f"{method.upper()} {path}"
            assert op.get("summary"), f"{where} has no summary"
            assert "200" in op["responses"], f"{where} has no 200 response"
            assert op["responses"]["200"].get("content"), f"{where} 200 has no content"


def test_security_schemes_are_declared():
    spec = _v1_spec()

    assert set(spec["components"]["securitySchemes"]) == {"ApiKeyBearer", "ApiKeyHeader"}
    assert spec["security"] == [{"ApiKeyBearer": []}, {"ApiKeyHeader": []}]


def test_undocumented_v1_route_fails_generation():
    """A new /v1 route without an overlay must fail loudly, not ship undocumented."""
    stub = _StubApp(
        {
            "openapi": "3.1.0",
            "info": {"version": "0.0.0"},
            "paths": {"/v1/brand-new": {"post": {}}},
        }
    )

    with pytest.raises(RuntimeError, match="brand-new"):
        build_v1_spec(stub)


def test_committed_spec_is_up_to_date():
    committed = DEFAULT_SPEC_PATH.read_text() if DEFAULT_SPEC_PATH.exists() else ""

    assert committed == serialize_spec(build_v1_spec()), (
        "docs/public/v1-openapi.json is stale; run `uv run llm-proxy-openapi` and commit it."
    )
