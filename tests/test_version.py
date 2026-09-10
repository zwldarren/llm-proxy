"""Tests for version reporting: plain release version vs. git-decorated display."""

import re
import subprocess
from pathlib import Path

import pytest

from llm_proxy import version as version_module
from llm_proxy.version import get_display_version


@pytest.fixture(autouse=True)
def _clear_version_caches():
    """Version functions cache their result process-wide; isolate each test."""
    version_module.get_version.cache_clear()
    version_module.get_display_version.cache_clear()
    yield
    version_module.get_version.cache_clear()
    version_module.get_display_version.cache_clear()


def _stub_git(monkeypatch: pytest.MonkeyPatch, describe: str | None) -> None:
    monkeypatch.setattr(version_module, "get_version", lambda: "0.2.1")
    monkeypatch.setattr(version_module, "_git_describe", lambda: describe)


class TestDisplayVersionParsing:
    def test_outside_git_checkout_reports_plain_version(self, monkeypatch):
        _stub_git(monkeypatch, None)

        assert get_display_version() == "0.2.1"

    def test_exact_clean_tag_reports_plain_version(self, monkeypatch):
        _stub_git(monkeypatch, "v0.2.1")

        assert get_display_version() == "0.2.1"

    def test_dirty_tagged_checkout_is_flagged(self, monkeypatch):
        _stub_git(monkeypatch, "v0.2.1-dirty")

        assert get_display_version() == "0.2.1-dirty"

    def test_commits_ahead_of_tag_are_named(self, monkeypatch):
        _stub_git(monkeypatch, "v0.2.1-13-gd64afc23d")

        assert get_display_version() == "0.2.1-13-gd64afc23d"

    def test_dirty_untagged_checkout_is_flagged(self, monkeypatch):
        _stub_git(monkeypatch, "v0.2.1-13-gd64afc23d-dirty")

        assert get_display_version() == "0.2.1-13-gd64afc23d-dirty"

    def test_tagless_checkout_keeps_base_version(self, monkeypatch):
        """A bare short sha (no tags reachable) is prefixed with the base version."""
        _stub_git(monkeypatch, "d64afc23d")

        assert get_display_version() == "0.2.1-d64afc23d"

    def test_tagless_dirty_checkout_keeps_base_version(self, monkeypatch):
        _stub_git(monkeypatch, "d64afc23d-dirty")

        assert get_display_version() == "0.2.1-d64afc23d-dirty"

    def test_untagged_head_reports_untagged_tag_version(self, monkeypatch):
        """A clean checkout exactly on a non-v-prefixed tag reports that tag."""
        _stub_git(monkeypatch, "1.2.3")

        assert get_display_version() == "1.2.3"


class TestGitDescribe:
    """Integration coverage for the subprocess wiring against real tmp repos."""

    @staticmethod
    def _init_repo(tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
        return repo

    def test_exact_tag_clean_tree(self, tmp_path, monkeypatch):
        repo = self._init_repo(tmp_path)
        _git(repo, "commit", "--allow-empty", "-m", "init")
        _git(repo, "tag", "v0.2.1")
        monkeypatch.setattr(version_module, "_REPO_ROOT", repo)

        assert version_module._git_describe() == "v0.2.1"

    def test_commit_after_tag(self, tmp_path, monkeypatch):
        repo = self._init_repo(tmp_path)
        _git(repo, "commit", "--allow-empty", "-m", "init")
        _git(repo, "tag", "v0.2.1")
        _git(repo, "commit", "--allow-empty", "-m", "next")
        monkeypatch.setattr(version_module, "_REPO_ROOT", repo)

        out = version_module._git_describe()

        assert out is not None and out.startswith("v0.2.1-1-g")

    def test_dirty_working_tree(self, tmp_path, monkeypatch):
        repo = self._init_repo(tmp_path)
        # --dirty only considers tracked modifications, not untracked files.
        (repo / "tracked.txt").write_text("initial")
        _git(repo, "add", "tracked.txt")
        _git(repo, "commit", "-m", "add tracked file")
        _git(repo, "tag", "v0.2.1")
        (repo / "tracked.txt").write_text("modified")
        monkeypatch.setattr(version_module, "_REPO_ROOT", repo)

        out = version_module._git_describe()

        assert out is not None and out.endswith("-dirty")

    def test_tagless_repo_falls_back_to_short_sha(self, tmp_path, monkeypatch):
        repo = self._init_repo(tmp_path)
        _git(repo, "commit", "--allow-empty", "-m", "init")
        monkeypatch.setattr(version_module, "_REPO_ROOT", repo)

        out = version_module._git_describe()

        assert out is not None and re.fullmatch(r"[0-9a-f]{7,40}", out)

    def test_non_git_directory_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_module, "_REPO_ROOT", tmp_path)

        assert version_module._git_describe() is None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
