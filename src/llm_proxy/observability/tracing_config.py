"""Tracing configuration for LLM observability."""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers import get_handler_type

logger = get_logger(__name__)


@dataclass
class TracingProviderConfig:
    """Configuration for a single tracing backend provider."""

    provider: str = "langfuse"
    name: str = "langfuse"
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "name": self.name,
            "enabled": self.enabled,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TracingProviderConfig:
        data = data or {}
        provider = data.get("provider", "langfuse")
        return cls(
            provider=provider,
            name=data.get("name", provider),
            enabled=data.get("enabled", True),
            settings=dict(data.get("settings") or {}),
        )

    @property
    def is_configured(self) -> bool:
        if not self.enabled:
            return False
        handler_cls = get_handler_type(self.provider)
        if handler_cls is None:
            return False
        return handler_cls.validate_config(self.settings)


@dataclass
class TracingConfig:
    """Configuration for tracing/observability."""

    enabled: bool = False
    providers: list[TracingProviderConfig] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        if not self.enabled or not self.providers:
            return False
        return any(p.is_configured for p in self.providers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "providers": [p.to_dict() for p in self.providers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TracingConfig:
        data = data or {}
        providers = [TracingProviderConfig.from_dict(p) for p in (data.get("providers") or [])]
        return cls(
            enabled=data.get("enabled", False),
            providers=providers,
        )


async def _shutdown_handler(handler: Any) -> None:
    """Gracefully shut down a tracing handler, awaiting async cleanup."""
    shutdown = getattr(handler, "shutdown", None)
    if shutdown is None:
        return
    try:
        await shutdown()
    except Exception as e:
        logger.debug(f"Error shutting down tracing handler: {e}")


__all__ = [
    "TracingConfig",
    "TracingProviderConfig",
]
