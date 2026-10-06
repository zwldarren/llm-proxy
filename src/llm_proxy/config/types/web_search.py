"""Web search tool configuration types."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_OLLAMA_BASE_URL = "https://ollama.com"


class SearXNGConfig(BaseModel):
    """SearXNG search provider configuration."""

    url: str = Field(..., description="SearXNG instance URL (e.g., http://localhost:8080)")
    api_key: str | None = Field(None, description="API key if required by the SearXNG instance")
    basic_auth_username: str | None = Field(
        None, description="Basic auth username for SearXNG instance"
    )
    basic_auth_password: str | None = Field(
        None, description="Basic auth password for SearXNG instance"
    )
    engines: list[str] | None = Field(
        None, description="List of search engines to use (e.g., ['google', 'bing'])"
    )
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=20, description="Maximum number of results to return")


class OllamaConfig(BaseModel):
    """Ollama web search provider configuration."""

    api_key: str = Field(
        ..., description="Ollama API key for web search (from https://ollama.com/settings/keys)"
    )
    base_url: str = Field(
        DEFAULT_OLLAMA_BASE_URL,
        description="Ollama API base URL",
    )
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=10, description="Maximum number of results to return")

    @field_validator("base_url")
    @classmethod
    def _default_blank_base_url(cls, value: str) -> str:
        """Use the hosted default when the field is blank or whitespace.

        The settings form binds this field directly, so clearing it stores an
        empty string; used verbatim that builds a relative request URL
        (``/api/web_search``) which fails before any network call.
        """
        return (value or "").strip() or DEFAULT_OLLAMA_BASE_URL


class WebSearchConfig(BaseModel):
    """Web search tool configuration.

    This configuration enables the proxy to intercept web_search tool calls
    and execute them via a search provider for providers that don't natively
    support web search (non-Anthropic providers).
    """

    enabled: bool = Field(False, description="Enable web search tool interception")
    provider: Literal["searxng", "ollama"] = Field("searxng", description="Search provider to use")
    searxng: SearXNGConfig | None = Field(
        None, description="SearXNG configuration (required if provider is 'searxng')"
    )
    ollama: OllamaConfig | None = Field(
        None, description="Ollama configuration (required if provider is 'ollama')"
    )

    @property
    def is_configured(self) -> bool:
        """Check if the web search is properly configured."""
        if not self.enabled:
            return False
        if self.provider == "searxng":
            return self.searxng is not None and bool(self.searxng.url)
        if self.provider == "ollama":
            return self.ollama is not None and bool(self.ollama.api_key)
        return False
