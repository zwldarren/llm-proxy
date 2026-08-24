"""Tests for the model pricing sync endpoint (api/routers/config/pricing.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from llm_proxy.api.routers.config import pricing
from llm_proxy.api.routers.config.pricing import (
    _fetch_models_dev_pricing,
    _find_model_pricing,
    _is_valid_pricing,
    apply_model_pricing,
    sync_model_pricing,
)
from llm_proxy.api.routers.config.pricing_schemas import (
    ApplyPricingRequest,
    ModelPricingInfo,
    PricingUpdateItem,
    SyncPricingRequest,
)


class TestIsValidPricing:
    """Pricing is valid only when at least one cost component is set."""

    def test_both_missing_is_invalid(self):
        assert not _is_valid_pricing(None, None)

    def test_both_zero_is_invalid(self):
        assert not _is_valid_pricing(0.0, 0.0)

    def test_input_only_is_valid(self):
        assert _is_valid_pricing(2.5, None)

    def test_output_only_is_valid(self):
        assert _is_valid_pricing(None, 10.0)

    def test_both_set_is_valid(self):
        assert _is_valid_pricing(2.5, 10.0)


class TestFindModelPricing:
    """Selecting the best pricing option for a model id."""

    def test_no_options_returns_none(self):
        selected, options = _find_model_pricing("gpt-4o", {})
        assert selected is None
        assert options == []

    def test_single_option_selected(self):
        info = ModelPricingInfo(model_id="gpt-4o", provider="openai", input_cost_per_1m=2.5)
        selected, options = _find_model_pricing("gpt-4o", {"gpt-4o": [info]})
        assert selected is info
        assert options == [info]

    def test_picks_lowest_provider_when_no_preference(self):
        openai = ModelPricingInfo(model_id="m", provider="openai", input_cost_per_1m=5.0)
        azure = ModelPricingInfo(model_id="m", provider="azure", input_cost_per_1m=3.0)
        selected, options = _find_model_pricing("m", {"m": [openai, azure]})
        assert selected is azure
        assert len(options) == 2

    def test_preferred_source_is_respected(self):
        openai = ModelPricingInfo(model_id="m", provider="openai", input_cost_per_1m=5.0)
        azure = ModelPricingInfo(model_id="m", provider="azure", input_cost_per_1m=3.0)
        selected, _ = _find_model_pricing("m", {"m": [openai, azure]}, preferred_source="openai")
        assert selected is openai


class TestFetchModelsDevPricing:
    """Parsing the nested models.dev payload into a model-id -> pricing map."""

    def _sample_payload(self) -> dict:
        return {
            "openai": {
                "models": {
                    "gpt-4o": {"cost": {"input": 2.5, "output": 10.0}},
                    "gpt-4o-mini": {"cost": {"input": 0.15, "output": 0.6}},
                }
            },
            "anthropic": {
                "models": {
                    "claude-3-5-sonnet": {"cost": {"input": 3.0, "output": 15.0}},
                }
            },
        }

    async def test_parses_nested_models(self):
        async def fake_fetch_json(client, url):
            return self._sample_payload()

        with patch.object(pricing, "fetch_json", fake_fetch_json):
            data = await _fetch_models_dev_pricing(MagicMock())

        assert set(data.keys()) == {"gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"}
        gpt4o = data["gpt-4o"][0]
        assert gpt4o.provider == "openai"
        assert gpt4o.input_cost_per_1m == 2.5
        assert gpt4o.output_cost_per_1m == 10.0

    async def test_skips_zero_or_missing_pricing(self):
        async def fake_fetch_json(client, url):
            return {
                "openai": {
                    "models": {
                        "free-model": {"cost": {"input": 0, "output": 0}},
                        "no-cost": {"cost": {}},
                        "good": {"cost": {"input": 1.0, "output": 2.0}},
                    }
                }
            }

        with patch.object(pricing, "fetch_json", fake_fetch_json):
            data = await _fetch_models_dev_pricing(MagicMock())

        assert "free-model" not in data
        assert "no-cost" not in data
        assert "good" in data

    async def test_skips_non_dict_provider_blocks(self):
        async def fake_fetch_json(client, url):
            return {"openai": "not-a-dict", "anthropic": {"models": "nope"}}

        with patch.object(pricing, "fetch_json", fake_fetch_json):
            data = await _fetch_models_dev_pricing(MagicMock())

        assert data == {}

    async def test_indexes_by_suffix_when_slash_present(self):
        async def fake_fetch_json(client, url):
            return {
                "openai": {
                    "models": {
                        "org/gpt-4o": {"cost": {"input": 1.0, "output": 2.0}},
                    }
                }
            }

        with patch.object(pricing, "fetch_json", fake_fetch_json):
            data = await _fetch_models_dev_pricing(MagicMock())

        assert "org/gpt-4o" in data
        assert "gpt-4o" in data


class TestSyncModelPricingEndpoint:
    """Behavior of the POST /models/sync-pricing endpoint."""

    @pytest.fixture
    def request_obj(self):
        return MagicMock(name="FastAPIRequest")

    @pytest.fixture
    def pricing_data(self):
        return {
            "gpt-4o": [
                ModelPricingInfo(
                    model_id="gpt-4o",
                    provider="openai",
                    input_cost_per_1m=2.5,
                    output_cost_per_1m=10.0,
                )
            ]
        }

    def _fake_repo(self, models):
        repo = MagicMock()
        repo.get_all_models = AsyncMock(return_value=models)
        repo._models.update_model_provider_pricing = AsyncMock()
        return repo

    def _model(self, name, mappings):
        model = MagicMock()
        model.name = name
        model.provider_mappings = mappings
        return model

    def _mapping(
        self,
        mid,
        provider_model_name,
        provider_name,
        input_cost=None,
        output_cost=None,
    ):
        mapping = MagicMock()
        mapping.id = mid
        mapping.provider_model_name = provider_model_name
        mapping.provider = MagicMock(name=provider_name)
        mapping.provider.name = provider_name
        mapping.input_cost_per_1m = input_cost
        mapping.output_cost_per_1m = output_cost
        mapping.cached_read_cost_per_1m = None
        mapping.cached_write_cost_per_1m = None
        mapping.audio_input_cost_per_1m = None
        mapping.audio_output_cost_per_1m = None
        return mapping

    async def test_http_status_error_returns_failure(self, request_obj):
        async def boom(client, url):
            raise httpx2.HTTPStatusError(
                "bad", request=MagicMock(), response=MagicMock(status_code=500)
            )

        with patch.object(pricing, "fetch_json", boom):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=True), request_obj, session=MagicMock()
            )

        assert not response.success
        assert "models.dev" in (response.error or "")

    async def test_request_error_returns_failure(self, request_obj):
        async def boom(client, url):
            raise httpx2.RequestError("network down")

        with patch.object(pricing, "fetch_json", boom):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=True), request_obj, session=MagicMock()
            )

        assert not response.success
        assert "Network error" in (response.error or "")

    async def test_dry_run_reports_pending_updates(self, request_obj):
        mapping = self._mapping(10, "gpt-4o", "openai", input_cost=None, output_cost=None)
        model = self._model("gpt-4o", [mapping])
        repo = self._fake_repo([model])

        async def fake_fetch_json(client, url):
            return {"openai": {"models": {"gpt-4o": {"cost": {"input": 2.5, "output": 10.0}}}}}

        with (
            patch.object(pricing, "fetch_json", fake_fetch_json),
            patch.object(pricing, "get_config_repository", return_value=repo),
        ):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=True), request_obj, session=MagicMock()
            )

        assert response.success
        assert response.dry_run
        assert response.updated_count == 1
        assert response.total_models == 1
        assert response.total_provider_mappings == 1
        assert not response.results[0].updated
        assert response.results[0].message == "Would update"
        # Dry run must not touch the database.
        repo._models.update_model_provider_pricing.assert_not_awaited()

    async def test_actual_run_updates_pricing_and_reloads(self, request_obj):
        mapping = self._mapping(10, "gpt-4o", "openai", input_cost=None, output_cost=None)
        model = self._model("gpt-4o", [mapping])
        repo = self._fake_repo([model])

        async def fake_fetch_json(client, url):
            return {"openai": {"models": {"gpt-4o": {"cost": {"input": 2.5, "output": 10.0}}}}}

        with (
            patch.object(pricing, "fetch_json", fake_fetch_json),
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=False), request_obj, session=MagicMock()
            )

        assert response.updated_count == 1
        assert response.results[0].updated
        repo._models.update_model_provider_pricing.assert_awaited_once_with(
            mapping.id,
            input_cost_per_1m=2.5,
            output_cost_per_1m=10.0,
            cached_read_cost_per_1m=None,
            cached_write_cost_per_1m=None,
            audio_input_cost_per_1m=None,
            audio_output_cost_per_1m=None,
        )
        reload_mock.assert_awaited_once()

    async def test_no_pricing_found_is_skipped(self, request_obj):
        mapping = self._mapping(11, "unknown-model", "openai")
        model = self._model("unknown-model", [mapping])
        repo = self._fake_repo([model])

        async def fake_fetch_json(client, url):
            return {}

        with (
            patch.object(pricing, "fetch_json", fake_fetch_json),
            patch.object(pricing, "get_config_repository", return_value=repo),
        ):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=True), request_obj, session=MagicMock()
            )

        assert response.skipped_count == 1
        assert response.updated_count == 0
        assert response.results[0].message == "No pricing found in models.dev"

    async def test_preserve_custom_pricing_skips_mapping(self, request_obj, pricing_data):
        mapping = self._mapping(12, "gpt-4o", "openai", input_cost=99.0, output_cost=199.0)
        model = self._model("gpt-4o", [mapping])
        repo = self._fake_repo([model])

        async def fake_fetch_json(client, url):
            return pricing_data

        with (
            patch.object(pricing, "fetch_json", fake_fetch_json),
            patch.object(pricing, "get_config_repository", return_value=repo),
        ):
            response = await sync_model_pricing(
                SyncPricingRequest(dry_run=False, preserve_custom_pricing=True),
                request_obj,
                session=MagicMock(),
            )

        assert response.skipped_count == 1
        assert response.updated_count == 0
        assert "preserved" in response.results[0].message
        repo._models.update_model_provider_pricing.assert_not_awaited()


class TestApplyModelPricingEndpoint:
    """Behavior of the POST /models/pricing/apply endpoint."""

    @pytest.fixture
    def request_obj(self):
        return MagicMock(name="FastAPIRequest")

    def _fake_repo(self):
        repo = MagicMock()
        repo._models.apply_mapping_pricing = AsyncMock(return_value=MagicMock())
        return repo

    async def test_applies_only_provided_fields(self, request_obj):
        repo = self._fake_repo()

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_pricing(
                ApplyPricingRequest(
                    updates=[
                        PricingUpdateItem(
                            mapping_id=10, input_cost_per_1m=2.5, output_cost_per_1m=10.0
                        )
                    ]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert response.success
        assert response.applied_count == 1
        assert response.failed_count == 0
        # Only explicitly provided fields are forwarded (exclude_unset semantics).
        repo._models.apply_mapping_pricing.assert_awaited_once_with(
            10, {"input_cost_per_1m": 2.5, "output_cost_per_1m": 10.0}
        )
        reload_mock.assert_awaited_once()

    async def test_explicit_null_clears_field(self, request_obj):
        repo = self._fake_repo()

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()),
        ):
            await apply_model_pricing(
                ApplyPricingRequest(
                    updates=[PricingUpdateItem(mapping_id=10, cached_read_cost_per_1m=None)]
                ),
                request_obj,
                session=MagicMock(),
            )

        repo._models.apply_mapping_pricing.assert_awaited_once_with(
            10, {"cached_read_cost_per_1m": None}
        )

    async def test_item_without_fields_fails(self, request_obj):
        repo = self._fake_repo()

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_pricing(
                ApplyPricingRequest(updates=[PricingUpdateItem(mapping_id=10)]),
                request_obj,
                session=MagicMock(),
            )

        assert not response.success
        assert response.applied_count == 0
        assert response.failed_count == 1
        assert response.results[0].message == "No pricing fields provided"
        repo._models.apply_mapping_pricing.assert_not_awaited()
        reload_mock.assert_not_awaited()

    async def test_missing_mapping_fails_others_still_apply(self, request_obj):
        repo = self._fake_repo()
        repo._models.apply_mapping_pricing = AsyncMock(side_effect=[None, MagicMock()])

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_pricing(
                ApplyPricingRequest(
                    updates=[
                        PricingUpdateItem(mapping_id=99, input_cost_per_1m=1.0),
                        PricingUpdateItem(mapping_id=10, input_cost_per_1m=2.5),
                    ]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert not response.success
        assert response.applied_count == 1
        assert response.failed_count == 1
        assert response.results[0].message == "Mapping not found"
        assert response.results[1].applied
        reload_mock.assert_awaited_once()

    async def test_applies_unit_based_pricing_fields(self, request_obj):
        """The apply endpoint accepts the unit-based pricing dimensions
        (per-image, per-audio-minute, per-1M-TTS-chars, per-1k-web-search,
        and per-1M-image-input-tokens) and forwards them to the repository."""
        repo = self._fake_repo()

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_pricing(
                ApplyPricingRequest(
                    updates=[
                        PricingUpdateItem(
                            mapping_id=10,
                            cost_per_image=0.04,
                            audio_cost_per_minute=0.006,
                            tts_cost_per_1m_chars=16.0,
                            web_search_cost_per_1k=0.03,
                            image_input_cost_per_1m=30.0,
                        )
                    ]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert response.success
        assert response.applied_count == 1
        repo._models.apply_mapping_pricing.assert_awaited_once_with(
            10,
            {
                "cost_per_image": 0.04,
                "audio_cost_per_minute": 0.006,
                "tts_cost_per_1m_chars": 16.0,
                "web_search_cost_per_1k": 0.03,
                "image_input_cost_per_1m": 30.0,
            },
        )
        reload_mock.assert_awaited_once()

    async def test_explicit_null_clears_unit_based_field(self, request_obj):
        """A provided null clears a unit-based pricing field."""
        repo = self._fake_repo()

        with (
            patch.object(pricing, "get_config_repository", return_value=repo),
            patch.object(pricing, "commit_and_reload", AsyncMock()),
        ):
            await apply_model_pricing(
                ApplyPricingRequest(
                    updates=[PricingUpdateItem(mapping_id=10, cost_per_image=None)]
                ),
                request_obj,
                session=MagicMock(),
            )

        repo._models.apply_mapping_pricing.assert_awaited_once_with(10, {"cost_per_image": None})
