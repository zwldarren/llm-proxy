"""Single source of truth for the llm-proxy package version.

``get_version`` is the plain release version (PEP 440) and the comparison
key for update checks. ``get_display_version`` decorates it with the exact
git commit whenever the running checkout is not an exact, clean tagged
release.

The decorated form is for human display only and is NOT PEP 440 — never
feed it to ``packaging.version`` (e.g. ``Version("0.2.1-13-gabc-dirty")``
raises ``InvalidVersion``); ``get_version`` is the only parseable form.
"""

import re
import shutil
import subprocess
import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Final

_DIST_NAME: Final[str] = "llm-proxy"
_FALLBACK_VERSION: Final[str] = "0.0.0+unknown"
#: Source-tree root. Outside a checkout (wheel install) git decoration is skipped.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
#: Bare ``git describe --always`` output: short/full sha, optionally "-dirty".
_BARE_SHA: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{7,40}(?:-dirty)?")


def _read_pyproject_version() -> str:
    """Read the static ``[project] version`` from the repo's pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        version = data["project"]["version"]
    except OSError, KeyError, tomllib.TOMLDecodeError:
        return _FALLBACK_VERSION
    return version if isinstance(version, str) else _FALLBACK_VERSION


def _git_describe() -> str | None:
    """Describe HEAD in the source tree, or None outside a git checkout.

    Output forms: ``v0.2.1`` on an exact clean tag, ``v0.2.1-dirty`` with
    local changes, and ``v0.2.1-<n>-g<sha>[-dirty]`` when HEAD is ahead of
    the tag. Falls back to a bare short sha when no tag is reachable (e.g.
    shallow clones). None when git is missing, the tree is not a checkout,
    or git fails.
    """
    if shutil.which("git") is None or not (_REPO_ROOT / ".git").exists():
        return None
    for args in (
        ["describe", "--tags", "--dirty"],
        # Shallow clones and tag-less checkouts: fall back to the short sha.
        ["describe", "--always", "--dirty"],
    ):
        try:
            result = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except OSError, subprocess.SubprocessError:
            continue
        if out := result.stdout.strip():
            return out
    return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the installed distribution version, cached.

    Falls back to the static version in pyproject.toml when the distribution
    metadata is unavailable (e.g. running from an uninstalled source tree),
    and to a placeholder when that fails too.

    Always plain PEP 440: this is the comparison key for update checks.
    """
    try:
        return _distribution_version(_DIST_NAME)
    except PackageNotFoundError:
        return _read_pyproject_version()


@lru_cache(maxsize=1)
def get_display_version() -> str:
    """Return the human-facing version, naming the exact commit whenever the
    running checkout is not an exact, clean tagged release.

    Builds on ``git describe``: a HEAD ahead of the latest tag yields
    ``<tag>-<n>-g<sha>``, with ``-dirty`` appended when the working tree has
    uncommitted changes, e.g. ``0.2.1-13-gd64afc23d-dirty``. An exact tagged
    checkout reports the plain release version, matching what a wheel
    install reports.

    The result is a display string, not PEP 440; only ``get_version`` is
    safe to parse or compare.
    """
    base = get_version()
    describe = _git_describe()
    if describe is None:
        return base
    if _BARE_SHA.fullmatch(describe):
        # No tag reachable (e.g. shallow clone): "<base>-<sha>[-dirty]".
        return f"{base}-{describe}"
    tag, sep, rest = describe.partition("-")
    if not sep:
        # Exactly on a clean tag, e.g. "v0.2.1".
        return tag.removeprefix("v")
    # Ahead of the tag and/or dirty, e.g. "v0.2.1-13-g<sha>[-dirty]".
    return f"{tag.removeprefix('v')}-{rest}"
