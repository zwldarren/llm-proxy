"""Provider serializer registry.

Separate registry from protocol serializers. Provider adapters use
get_provider_serializer() to obtain the serializer for their provider.

Thin facade over CachedRegistry (core/registry_base); public API unchanged.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

from llm_proxy.core.registry_base import CachedRegistry
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.serialization.providers.base import ProviderSerializer

logger = get_logger(__name__)

ProviderSerializerClass = type["ProviderSerializer"]


def _import_locations(name: str) -> tuple[str, str]:
    """On-demand import candidates for a provider serializer.

    Family dialect serializers live next to their dialect knowledge
    (ADR-0003); provider-local ones stay in providers/.
    """
    return (
        f"llm_proxy.providers.{name}.serializer",
        f"llm_proxy.serialization.{name}.serializer",
    )


def _post_create(instance: ProviderSerializer, name: str) -> None:
    instance._registered_provider_name = name


_serializers = CachedRegistry["ProviderSerializer"](
    label="provider serializer",
    import_locations=_import_locations,
    post_create=_post_create,
)


@overload
def register_provider_serializer(
    name: str,
    serializer_cls: ProviderSerializerClass,
) -> ProviderSerializerClass: ...


@overload
def register_provider_serializer(
    name: str,
    serializer_cls: None = None,
) -> Callable[[ProviderSerializerClass], ProviderSerializerClass]: ...


def register_provider_serializer(
    name: str,
    serializer_cls: ProviderSerializerClass | None = None,
) -> Callable[[ProviderSerializerClass], ProviderSerializerClass] | ProviderSerializerClass:
    """Decorator to register a provider serializer class.

    Args:
        name: Provider name (e.g., "openai", "anthropic", "gemini")
        serializer_cls: The serializer class to register (optional)

    Example:
        @register_provider_serializer("openai")
        class OpenAIProviderSerializer(ProviderSerializer):
            ...
    """

    def decorator(cls: ProviderSerializerClass) -> ProviderSerializerClass:
        _serializers.register(name, cls)
        logger.debug(
            f"Registered provider serializer '{_serializers.canonical(name)}' "
            f"from {cls.__module__}.{cls.__name__}"
        )
        return cls

    if serializer_cls is not None:
        return decorator(serializer_cls)
    return decorator


def get_provider_serializer(name: str) -> ProviderSerializer:
    """Get a provider serializer instance by name.

    Instances are cached — serializers are stateless so reuse is safe.

    Auto-discovery: if no serializer is registered for *name* yet, the registry
    imports ``llm_proxy.providers.{name}.serializer`` then
    ``llm_proxy.serialization.{name}.serializer`` on demand. This runs the
    ``@register_provider_serializer`` decorator in that module, registering
    the class before we look it up.

    Provider names are case-insensitive; the canonical lowercase form is used for
    lookups and imports.

    Args:
        name: Provider name (e.g., "openai", "anthropic", "gemini", "Gemini")

    Returns:
        ProviderSerializer instance

    Raises:
        ConfigurationError: If no serializer is registered for the provider
    """
    return _serializers.get(name)


__all__ = [
    "register_provider_serializer",
    "get_provider_serializer",
]
