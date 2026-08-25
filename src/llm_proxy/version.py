"""Single source of truth for the llm-proxy package version."""

import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Final

_DIST_NAME: Final[str] = "llm-proxy"
_FALLBACK_VERSION: Final[str] = "0.0.0+unknown"


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


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the installed distribution version, cached.

    Falls back to the static version in pyproject.toml when the distribution
    metadata is unavailable (e.g. running from an uninstalled source tree),
    and to a placeholder when that fails too.
    """
    try:
        return _distribution_version(_DIST_NAME)
    except PackageNotFoundError:
        return _read_pyproject_version()
