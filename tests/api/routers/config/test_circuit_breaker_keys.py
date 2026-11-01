"""Contract test: provider mapping keys stay parseable by the circuit-breaker API.

``core/provider_selector.py`` builds the key (``provider:model:index``) that the
breaker store, the latency store and the Redis sticky mapping are all keyed by;
``api/routers/config/circuit_breaker.py`` parses it back apart for the console.
The selector derives the index from its own canonical enumeration, so the parse
must not be sensitive to how a provider or model name is spelled.
"""

import pytest

from llm_proxy.api.routers.config.circuit_breaker import _parse_provider_key
from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
from llm_proxy.core.provider_selector import ProviderSelector


def _selector(providers: list[ModelProviderConfig]) -> ProviderSelector:
    return ProviderSelector(
        model_config=ModelConfig(providers=providers, model_name="test-model"),
        provider_configs={},
    )


@pytest.mark.parametrize(
    ("provider", "upstream_model"),
    [
        ("openai", None),
        ("openrouter", "anthropic/claude-sonnet-4"),
        ("opencode-go", "glm-5"),
    ],
)
def test_mapping_key_round_trips_through_the_api_parser(provider, upstream_model):
    """The selector's key splits back into exactly the provider and model it names."""
    selector = _selector(
        [ModelProviderConfig(provider=provider, provider_model_name=upstream_model, priority=1)]
    )
    group = selector._get_providers_by_priority_with_keys()[0]
    _, key = group[0]

    parsed_provider, parsed_model, parsed_index = _parse_provider_key(key)

    assert parsed_provider == provider
    assert parsed_model == (upstream_model or "")
    assert parsed_index == 0


def test_mapping_key_is_stable_for_a_reloaded_config():
    """Loading the same mappings again yields the same keys (state stays attached)."""
    providers = [
        ModelProviderConfig(provider="a", priority=1),
        ModelProviderConfig(provider="b", priority=1),
    ]

    first = [
        key
        for group in _selector(providers)._get_providers_by_priority_with_keys()
        for _, key in group
    ]
    second = [
        key
        for group in _selector(list(providers))._get_providers_by_priority_with_keys()
        for _, key in group
    ]

    assert first == second
