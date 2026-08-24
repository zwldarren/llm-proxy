"""Thread-safe registry base class.

This module provides a thread-safe registry implementation that can be used
as a base class for any registry that needs concurrent access protection.
"""

import importlib
from collections.abc import Callable, Iterable
from threading import RLock
from typing import TypeVar

from llm_proxy.core.exceptions import ConfigurationError

T = TypeVar("T")


class ThreadSafeRegistry[T]:
    """Thread-safe registry base class.

    This class provides a thread-safe implementation for storing and retrieving
    values by name. It uses an RLock (reentrant lock) to protect all operations
    on the underlying dictionary.

    Example:
        ```python
        registry = ThreadSafeRegistry[str]()
        registry.register("provider1", "value1")
        value = registry.get("provider1")  # Returns "value1"
        names = registry.list_all()  # Returns ["provider1"]
        ```
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty dictionary and lock."""
        self._registry: dict[str, T] = {}
        self._lock = RLock()

    def register(self, name: str, value: T) -> None:
        """Register a value with the given name.

        Thread-safe operation that stores the value in the registry.

        Args:
            name: The name to register the value under
            value: The value to register
        """
        with self._lock:
            self._registry[name] = value

    def get(self, name: str) -> T | None:
        """Get a value by name.

        Thread-safe operation that retrieves the value from the registry.

        Args:
            name: The name to look up

        Returns:
            The registered value, or None if not found
        """
        with self._lock:
            return self._registry.get(name)

    def list_all(self) -> list[str]:
        """List all registered names.

        Thread-safe operation that returns a list of all registered names.

        Returns:
            List of all registered names
        """
        with self._lock:
            return list(self._registry.keys())

    def get_all(self) -> dict[str, T]:
        """Get a copy of the entire registry.

        Thread-safe operation that returns a copy of the registry dictionary.

        Returns:
            A copy of the registry dictionary
        """
        with self._lock:
            return dict(self._registry)

    def clear(self) -> None:
        """Clear all entries from the registry.

        Thread-safe operation that removes all entries.
        """
        with self._lock:
            self._registry.clear()

    def __len__(self) -> int:
        """Get the number of registered entries.

        Thread-safe operation.

        Returns:
            The number of registered entries
        """
        with self._lock:
            return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered using 'in' operator.

        Thread-safe operation.

        Args:
            name: The name to check

        Returns:
            True if the name is registered, False otherwise
        """
        with self._lock:
            return name in self._registry


class CachedRegistry[I]:
    """Thread-safe class registry with singleton instance caching.

    Shared machinery for the serializer registries (protocol + provider):
    classes register under canonical lowercase names; instances are created
    lazily, cached, and reused (serializers are stateless). On a registry
    miss, optional *import_locations* are imported in order to trigger
    decorator-based registration.

    The instance lock is reentrant: on-demand imports run module-level code,
    and adapter modules call back into ``get()`` at import time — a plain
    Lock would self-deadlock the same thread.

    Args:
        label: Human label used in miss error messages (e.g. "provider
            serializer").
        import_locations: Optional callable mapping a canonical name to
            candidate module paths tried on registry miss.
        post_create: Optional hook invoked as ``hook(instance, name)`` after
            each instance creation.
    """

    def __init__(
        self,
        *,
        label: str,
        import_locations: Callable[[str], Iterable[str]] | None = None,
        post_create: Callable[[I, str], None] | None = None,
    ) -> None:
        self._label = label
        self._classes: ThreadSafeRegistry[type[I]] = ThreadSafeRegistry()
        self._instances: dict[str, I] = {}
        self._lock = RLock()
        self._import_locations = import_locations
        self._post_create = post_create

    @staticmethod
    def canonical(name: str) -> str:
        """Return the canonical (lowercase) form of a registry key."""
        return name.lower()

    def register(self, name: str, cls: type[I]) -> type[I]:
        """Register a class under the canonical name; returns it unchanged."""
        self._classes.register(self.canonical(name), cls)
        return cls

    def get_class(self, name: str) -> type[I] | None:
        """Return the registered class for *name*, or None."""
        return self._classes.get(self.canonical(name))

    def list_all(self) -> list[str]:
        """List all registered names."""
        return self._classes.list_all()

    def get(self, name: str) -> I:
        """Return the cached singleton instance for *name*.

        On a registry miss the optional import locations are tried (each
        import runs the module's registration decorator). Raises
        ConfigurationError when nothing is registered.
        """
        canonical = self.canonical(name)
        cached = self._instances.get(canonical)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._instances.get(canonical)
            if cached is not None:
                return cached

            cls = self._classes.get(canonical)
            imported = False
            if cls is None and self._import_locations is not None:
                for location in self._import_locations(canonical):
                    try:
                        importlib.import_module(location)
                        imported = True
                    except ModuleNotFoundError:
                        continue
                    cls = self._classes.get(canonical)
                    if cls is not None:
                        break

            if cls is None:
                available = ", ".join(sorted(self.list_all())) or "none"
                if imported:
                    raise ConfigurationError(
                        message=f"Serializer module for '{name}' exists but did not "
                        f"register any {self._label}. Available: {available}"
                    )
                raise ConfigurationError(
                    message=f"No {self._label} registered for '{name}'. Available: {available}"
                )

            instance = cls()
            if self._post_create is not None:
                self._post_create(instance, canonical)
            self._instances[canonical] = instance
            return instance
