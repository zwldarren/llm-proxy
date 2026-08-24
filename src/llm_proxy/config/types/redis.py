"""Redis configuration types."""

from pydantic import BaseModel, Field


class RedisRateLimitConfig(BaseModel):
    """Redis rate limiting configuration."""

    enabled: bool = Field(default=False, description="Enable Redis-based rate limiting")
    prefix: str = Field(default="rate_limit:", description="Redis key prefix for rate limiting")


class RedisCacheConfig(BaseModel):
    """Redis caching configuration."""

    enabled: bool = Field(default=False, description="Enable Redis caching")
    prefix: str = Field(default="cache:", description="Redis key prefix for caching")
    ttl_provider_config: int = Field(
        default=300, description="TTL in seconds for provider configuration cache"
    )
    ttl_model_mapping: int = Field(
        default=300, description="TTL in seconds for model mapping cache"
    )


class RedisLoggingConfig(BaseModel):
    """Redis logging configuration."""

    enabled: bool = Field(default=False, description="Enable Redis logging backend")
    prefix: str = Field(default="logs:", description="Redis key prefix for logging")
    ttl_days: int = Field(default=30, description="TTL in days for log entries")
    batch_size: int = Field(default=50, description="Batch size for log flushing")
    flush_interval_ms: int = Field(default=250, description="Flush interval in milliseconds")


class RedisConfig(BaseModel):
    """Redis configuration."""

    enabled: bool = Field(default=False, description="Enable Redis support")
    url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    pool_size: int = Field(default=10, description="Redis connection pool size")
    timeout: float = Field(default=5.0, description="Redis connection timeout in seconds")
    rate_limit: RedisRateLimitConfig = Field(
        default_factory=RedisRateLimitConfig,
        description="Redis rate limiting configuration",
    )
    cache: RedisCacheConfig = Field(
        default_factory=RedisCacheConfig,
        description="Redis caching configuration",
    )
    logging: RedisLoggingConfig = Field(
        default_factory=RedisLoggingConfig,
        description="Redis logging configuration",
    )
