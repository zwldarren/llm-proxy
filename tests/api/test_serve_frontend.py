"""Tests for the frontend SPA fallback route (serve_frontend)."""

from pathlib import Path

import pytest

from llm_proxy.api import resolve_frontend_file, serve_frontend_response
from llm_proxy.core.exceptions import NotFoundError


def test_resolve_frontend_file_allows_safe_path(tmp_path: Path) -> None:
    """A path inside the static root resolves successfully."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("ok")

    resolved = resolve_frontend_file(static_root, "index.html")
    assert resolved == (static_root / "index.html").resolve()


def test_resolve_frontend_file_blocks_traversal(tmp_path: Path) -> None:
    """Dot-dot segments that escape the static root are rejected."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "credentials.txt").write_text("secret")

    with pytest.raises(NotFoundError):
        resolve_frontend_file(static_root, "../secret/credentials.txt")

    with pytest.raises(NotFoundError):
        resolve_frontend_file(static_root, "../../secret/credentials.txt")


def test_resolve_frontend_file_blocks_nested_traversal(tmp_path: Path) -> None:
    """Dot-dot segments nested inside a sub-directory still escape detection."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "static").mkdir()
    (static_root / "static" / "style.css").write_text("body { color: red; }")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "credentials.txt").write_text("secret")

    # static/.. cancels to dist, so another .. escapes dist
    with pytest.raises(NotFoundError):
        resolve_frontend_file(static_root, "static/../../secret/credentials.txt")

    # Same with a deeper nested chain
    (static_root / "static" / "nested").mkdir()
    with pytest.raises(NotFoundError):
        resolve_frontend_file(static_root, "static/nested/../../../secret/credentials.txt")


def test_resolve_frontend_file_blocks_absolute_path(tmp_path: Path) -> None:
    """Absolute paths must not bypass the static root boundary."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("ok")

    with pytest.raises(NotFoundError):
        resolve_frontend_file(static_root, "/etc/passwd")


def test_resolve_frontend_file_allows_empty_path(tmp_path: Path) -> None:
    """Empty path string resolves safely to the static root boundary."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("ok")

    resolved = resolve_frontend_file(static_root, "")
    assert resolved == static_root.resolve()


def test_serve_frontend_response_serves_file(tmp_path: Path) -> None:
    """Existing static file inside dist is served."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>SPA</html>")
    (static_root / "style.css").write_text("body { color: red; }")

    response = serve_frontend_response(static_root, "style.css")
    assert response.path == (static_root / "style.css").resolve()


def test_serve_frontend_response_falls_back_to_index(tmp_path: Path) -> None:
    """Unknown routes fall back to index.html."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>SPA</html>")

    response = serve_frontend_response(static_root, "some/spa/route")
    assert response.path == (static_root / "index.html").resolve()


def test_serve_frontend_response_blocks_traversal(tmp_path: Path) -> None:
    """Traversal attempts are answered with NotFoundError."""
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>SPA</html>")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "credentials.txt").write_text("secret")

    with pytest.raises(NotFoundError):
        serve_frontend_response(static_root, "../secret/credentials.txt")


def test_serve_frontend_response_blocks_api_prefix(tmp_path: Path) -> None:
    """Paths starting with 'api/' should NOT fall back to index.html.

    SPA fallback is intended for client-side routes, but API paths are
    server-side endpoints that should return 404 when not found, rather
    than serving the SPA shell.
    """
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>SPA</html>")

    with pytest.raises(NotFoundError):
        serve_frontend_response(static_root, "api/nonexistent")
