"""Main proxy configuration type."""

from pydantic import BaseModel, Field

from llm_proxy.config.types.model import ModelConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.config.types.provider_selection import ProviderSelectionConfig
from llm_proxy.config.types.redis import RedisConfig
from llm_proxy.config.types.server import ServerParams
from llm_proxy.config.types.smart_routing import SmartRoutingConfig


class ProxyConfig(BaseModel):
    """Main configuration for LLM Proxy."""

    server_params: ServerParams = Field(
        default_factory=ServerParams,
        description="Server configuration parameters",
    )
    provider_configs: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Provider configurations",
    )
    models: dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Independent model configurations",
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis configuration",
    )
    smart_routing: SmartRoutingConfig = Field(
        default_factory=SmartRoutingConfig,
        description="Global smart routing configuration (server_config: smart_routing)",
    )
    provider_selection: ProviderSelectionConfig = Field(
        default_factory=ProviderSelectionConfig,
        description="Global provider-selection strategy (server_config: provider_selection)",
    )
