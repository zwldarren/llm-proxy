"""Auto-discovery of provider and serializer modules.

Replaces manual import lists with importlib-based auto-scanning.
"""

import importlib
import pkgutil
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType


def discover_packages(
    package_name: str, package_path: str, exclude: Iterable[str] = ()
) -> list[str]:
    """Find all sub-packages in a package directory.

    Returns list of package names (not modules) found.
    """
    path = Path(package_path)
    found = []
    for entry in sorted(path.iterdir()):
        if not entry.name.startswith("_") and entry.is_dir():
            init_file = entry / "__init__.py"
            if init_file.exists() and entry.name not in exclude:
                found.append(entry.name)
    return found


def import_all_from_package(
    package_name: str,
    submodule: str = "adapter",
    exclude: Iterable[str] = ("utils", "__pycache__"),
) -> list[ModuleType]:
    """Import a specific submodule from all sub-packages of a package.

    Traverses subdirectories of the given package and imports
    ``{package_name}.{sub_pkg}.{submodule}`` for each.

    Args:
        package_name: Full dotted package name (e.g. "llm_proxy.providers")
        submodule: Name of the module to import from each sub-package
        exclude: Sub-package names to skip

    Returns:
        List of successfully imported modules
    """
    imported = []

    # Only the package import itself is allowed to fail silently.
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return imported

    if not hasattr(package, "__path__"):
        return imported

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg and name not in exclude:
            try:
                mod = importlib.import_module(f"{package_name}.{name}.{submodule}")
                imported.append(mod)
            except ImportError:
                # Two tolerated cases: the submodule simply does not exist
                # (ModuleNotFoundError — most providers have no
                # ``serializer.py``), and the serialization↔providers
                # bootstrap cycle, where importing a provider package runs
                # its adapter module while ``providers/base.py`` is still
                # partially initialized. The later scan from
                # ``providers/__init__`` completes registration.
                # Any OTHER error (e.g. the capability-mixin MRO check in
                # BaseHttpProvider.__init_subclass__) propagates loudly —
                # otherwise one broken module would abort this loop and
                # silently drop every provider after it from the registry.
                continue

    return imported


__all__ = ["discover_packages", "import_all_from_package"]
