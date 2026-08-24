"""Protocol registry: protocol endpoints and protocol serializers.

Both protocol concepts register here:

- ``register_protocol`` — ProtocolEndpoint instances (HTTP route shapes),
  registered from ``protocols/<name>/__init__.py``.
- ``register_protocol_serializer`` — ProtocolSerializer classes (wire-format
  conversion), registered from ``protocols/<name>/serializer.py``.

Endpoints are stored on a plain ThreadSafeRegistry (objects, not classes);
serializer classes go through CachedRegistry which owns the class store,
singleton instance cache and double-checked locking (see
``core/registry_base``). Serializers are stateless, so instance reuse is safe.
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, overload

from llm_proxy.core.registry_base import CachedRegistry, ThreadSafeRegistry
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.base import ProtocolEndpoint

if TYPE_CHECKING:
    from llm_proxy.protocols.serializer_base import ProtocolSerializer

logger = get_logger(__name__)

ProtocolEndpointType = ProtocolEndpoint

_protocols = ThreadSafeRegistry[ProtocolEndpointType]()


@overload
def register_protocol(
    endpoint: ProtocolEndpoint, *, name: str | None = None
) -> ProtocolEndpoint: ...


@overload
def register_protocol(
    endpoint: None = None, *, name: str | None = None
) -> Callable[[ProtocolEndpoint], ProtocolEndpoint]: ...


def register_protocol(
    endpoint: ProtocolEndpoint | None = None,
    *,
    name: str | None = None,
) -> Callable[[ProtocolEndpoint], ProtocolEndpoint] | ProtocolEndpoint:
    """Decorator to register a protocol endpoint."""

    def decorator(ep: ProtocolEndpoint) -> ProtocolEndpoint:
        protocol_name = name or ep.name
        if _protocols.get(protocol_name) is not None:
            warnings.warn(
                f"Protocol '{protocol_name}' is being re-registered.",
                stacklevel=2,
            )
        _protocols.register(protocol_name, ep)
        logger.debug(f"Registered protocol '{protocol_name}'")
        return ep

    if endpoint is not None:
        return decorator(endpoint)
    return decorator


def get_protocol(name: str) -> ProtocolEndpoint | None:
    return _protocols.get(name)


def list_protocols() -> list[str]:
    return sorted(_protocols.list_all())


def get_protocols_info() -> list[dict[str, str]]:
    info = []
    for endpoint in _protocols.get_all().values():
        paths = endpoint.paths
        path = paths[0] if paths else ""
        info.append(
            {
                "name": endpoint.name,
                "path": path,
                "description": endpoint.description,
            }
        )
    return sorted(info, key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# Protocol serializer registry
# ---------------------------------------------------------------------------

ProtocolSerializerClass = type["ProtocolSerializer"]

_serializers = CachedRegistry["ProtocolSerializer"](label="protocol serializer")


@overload
def register_protocol_serializer(
    name: str,
    serializer_cls: ProtocolSerializerClass,
) -> ProtocolSerializerClass: ...


@overload
def register_protocol_serializer(
    name: str,
    serializer_cls: None = None,
) -> Callable[[ProtocolSerializerClass], ProtocolSerializerClass]: ...


def register_protocol_serializer(
    name: str,
    serializer_cls: ProtocolSerializerClass | None = None,
) -> Callable[[ProtocolSerializerClass], ProtocolSerializerClass] | ProtocolSerializerClass:
    """Decorator to register a protocol serializer class.

    Args:
        name: Protocol name (e.g., "openai", "anthropic", "openai_responses")
        serializer_cls: The serializer class to register (optional)

    Example:
        @register_protocol_serializer("openai")
        class OpenAIProtocolSerializer(ProtocolSerializer):
            ...
    """

    def decorator(cls: ProtocolSerializerClass) -> ProtocolSerializerClass:
        _serializers.register(name, cls)
        logger.debug(
            f"Registered protocol serializer '{_serializers.canonical(name)}' "
            f"from {cls.__module__}.{cls.__name__}"
        )
        return cls

    if serializer_cls is not None:
        return decorator(serializer_cls)
    return decorator


def get_protocol_serializer(name: str) -> ProtocolSerializer:
    """Get a protocol serializer instance by name.

    Instances are cached — serializers are stateless so reuse is safe.
    Protocol names are case-insensitive; the canonical lowercase form is used.

    Args:
        name: Protocol name (e.g., "openai", "anthropic", "openai_responses")

    Returns:
        ProtocolSerializer instance

    Raises:
        ConfigurationError: If no serializer is registered for the protocol
    """
    return _serializers.get(name)


__all__ = [
    "ProtocolEndpoint",
    "ProtocolSerializer",
    "get_protocol",
    "get_protocol_serializer",
    "get_protocols_info",
    "list_protocols",
    "register_protocol",
    "register_protocol_serializer",
]
