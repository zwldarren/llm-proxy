"""Catalog parsing, caching and lookup against the models.dev payload."""

import pytest

import llm_proxy.providers.opencode.catalog as catalog
from llm_proxy.providers.opencode.catalog import lookup, parse_catalog, peek
from llm_proxy.providers.opencode.dialect import Dialect


@pytest.fixture(autouse=True)
def _fresh_cache():
    catalog.reset_cache_for_tests()
    yield
    catalog.reset_cache_for_tests()


def _payload() -> dict:
    return {
        "opencode": {
            "api": "https://opencode.ai/inference/v1",
            "models": {
                "claude-opus-4-5": {
                    "provider": {"npm": "@ai-sdk/anthropic"},
                    "name": "Claude Opus 4.5",
                    "description": "coding model",
                    "limit": {"context": 200000, "output": 32000},
                    "cost": {"input": 5, "output": 25, "cache_read": 0.5},
                },
                "glm-5.3": {
                    "name": "GLM 5.3",
                    "limit": {"context": 128000},
                    "cost": {"input": 1.4, "output": 4.4},
                },
                "free-model": {"cost": {"input": 0, "output": 0}},
            },
        },
        "opencode-go": {
            "api": "https://opencode.ai/inference/go/v1",
            "models": {
                # Same id as the Console entry, different dialect: the
                # per-lane split that makes static tables rot.
                "minimax-m2.7": {"provider": {"npm": "@ai-sdk/anthropic"}},
            },
        },
        "openai": {"models": {"gpt-5": {}}},
    }


class TestParseCatalog:
    def test_keeps_only_the_opencode_entries(self):
        tables = parse_catalog(_payload())

        assert set(tables) == {"opencode", "opencode-go"}
        assert set(tables["opencode"]) == {"claude-opus-4-5", "glm-5.3", "free-model"}
        assert set(tables["opencode-go"]) == {"minimax-m2.7"}

    def test_npm_marker_becomes_the_dialect(self):
        tables = parse_catalog(_payload())

        assert lookup(tables, "opencode", "claude-opus-4-5").dialect is Dialect.ANTHROPIC
        assert lookup(tables, "opencode", "glm-5.3").dialect is Dialect.CHAT
        assert lookup(tables, "opencode-go", "minimax-m2.7").dialect is Dialect.ANTHROPIC

    def test_pricing_is_normalized_to_per_token(self):
        tables = parse_catalog(_payload())
        pricing = lookup(tables, "opencode", "claude-opus-4-5").pricing

        assert pricing.prompt == pytest.approx(5 / 1_000_000)
        assert pricing.completion == pytest.approx(25 / 1_000_000)

    def test_metadata_flows_through(self):
        tables = parse_catalog(_payload())
        entry = lookup(tables, "opencode", "claude-opus-4-5")

        assert entry.name == "Claude Opus 4.5"
        assert entry.context_length == 200000
        assert entry.description == "coding model"

    @pytest.mark.parametrize(
        "payload",
        [[], {}, {"openai": {"models": {}}}, {"opencode": {"models": []}}],
        ids=("not-an-object", "no-entries", "empty-models", "non-dict-models"),
    )
    def test_a_payload_without_opencode_entries_raises(self, payload):
        with pytest.raises(ValueError):
            parse_catalog(payload)


class TestLookup:
    def test_unknown_entry_or_model(self):
        tables = parse_catalog(_payload())

        assert lookup(tables, "opencode", "nope") is None
        assert lookup(tables, "nope", "claude-opus-4-5") is None
        assert lookup(tables, "opencode", None) is None
        assert lookup({}, "opencode", "glm-5.3") is None


class TestCache:
    def test_warm_cache_is_read_synchronously(self):
        catalog._cache = (catalog.time.monotonic(), parse_catalog(_payload()))

        assert set(peek()) == {"opencode", "opencode-go"}

    def test_cold_cache_is_empty(self):
        assert peek() == {}

    def test_stale_positive_cache_expires(self):
        catalog._cache = (catalog.time.monotonic() - catalog._POSITIVE_TTL_SECONDS - 1, {})

        # Stale reads as cold: an empty dict, so the sync resolution path
        # falls through to the heuristics.
        assert peek() == {}

    def test_negative_cache_within_the_window_suppresses(self):
        catalog._cache = (catalog.time.monotonic(), None)

        assert peek() == {}

    def test_negative_cache_expires(self):
        catalog._cache = (catalog.time.monotonic() - catalog._NEGATIVE_TTL_SECONDS - 1, None)

        assert peek() == {}

    async def test_load_fetches_once_and_caches(self, mock_response_cls, make_mock_client):
        client = make_mock_client(mock_response_cls(json_data=_payload()))

        first = await catalog.load_catalog(client)
        second = await catalog.load_catalog(client)

        assert first == second
        assert client.request.await_count == 1
        assert lookup(first, "opencode", "claude-opus-4-5").dialect is Dialect.ANTHROPIC

    async def test_fetch_failure_negatively_caches(self, mock_response_cls, make_mock_client):
        client = make_mock_client(
            mock_response_cls(status_code=500, text_data="boom"),
        )

        assert await catalog.load_catalog(client) == {}
        # The failure is cached for the negative window: no second fetch.
        assert await catalog.load_catalog(client) == {}
        assert client.request.await_count == 1

    async def test_single_flight_shares_one_fetch(self, mock_response_cls, make_mock_client):
        """Concurrent first requests share one fetch behind the lock."""
        import asyncio

        client = make_mock_client(mock_response_cls(json_data=_payload()))

        results = await asyncio.gather(
            catalog.load_catalog(client),
            catalog.load_catalog(client),
            catalog.load_catalog(client),
        )

        assert client.request.await_count == 1
        assert all(r == results[0] for r in results)
