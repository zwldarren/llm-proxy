"""Tests for the model metadata sync endpoints (api/routers/config/metadata.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from llm_proxy.api.routers.config import metadata
from llm_proxy.api.routers.config.metadata import (
    _changed_fields,
    _entry_to_option,
    apply_model_metadata,
    sync_model_metadata,
)
from llm_proxy.api.routers.config.metadata_schemas import (
    ApplyMetadataRequest,
    MetadataUpdateItem,
    ModelMetadataOption,
    SyncMetadataResponse,
)


def _catalog_entry(**overrides) -> dict:
    entry = {
        "attachment": True,
        "reasoning": True,
        "tool_call": False,
        "structured_output": True,
        "temperature": True,
        "open_weights": False,
        "status": "beta",
        "family": "gpt",
        "knowledge": "2025-01-01",
        "release_date": "2025-02-01",
        "limit": {"context": 128000, "output": 16384},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
    }
    entry.update(overrides)
    return entry


class TestEntryToOption:
    """Mapping a models.dev model entry onto a metadata option."""

    def test_maps_all_fields(self):
        option = _entry_to_option("openai", _catalog_entry())

        assert option.source == "openai"
        assert option.attachment is True
        assert option.tool_call is False
        assert option.status == "beta"
        assert option.family == "gpt"
        assert option.knowledge == "2025-01-01"
        assert option.release_date == "2025-02-01"
        assert option.context_length == 128000
        assert option.max_output_tokens == 16384
        assert option.supports_images is True

    def test_modalities_without_image_is_false(self):
        entry = _catalog_entry(modalities={"input": ["text"], "output": ["text"]})
        assert _entry_to_option("openai", entry).supports_images is False

    def test_missing_modalities_is_none(self):
        entry = _catalog_entry()
        del entry["modalities"]
        assert _entry_to_option("openai", entry).supports_images is None

    def test_non_bool_fields_without_data_are_none(self):
        entry = _catalog_entry(attachment="yes", limit={"context": "oops"})
        option = _entry_to_option("openai", entry)

        assert option.attachment is None
        assert option.context_length is None

    def test_non_dict_limit_and_modalities_are_ignored(self):
        option = _entry_to_option("openai", _catalog_entry(limit="big", modalities="many"))

        assert option.context_length is None
        assert option.max_output_tokens is None
        assert option.supports_images is None


class TestChangedFields:
    """Diffing stored values against a catalog candidate."""

    def test_no_change_when_equal(self):
        old = _entry_to_option("", _catalog_entry())
        new = _entry_to_option("openai", _catalog_entry())
        assert _changed_fields(old, new) == []

    def test_none_candidate_fields_never_change(self):
        old = _entry_to_option("", _catalog_entry())
        new = ModelMetadataOption(source="openai")  # everything None
        assert _changed_fields(old, new) == []

    def test_detects_bool_flip(self):
        old = _entry_to_option("", _catalog_entry(tool_call=False))
        new = ModelMetadataOption(source="openai", tool_call=True)
        assert _changed_fields(old, new) == ["tool_call"]


class TestSyncModelMetadataEndpoint:
    """Behavior of the POST /models/sync-metadata endpoint."""

    @pytest.fixture
    def request_obj(self):
        return MagicMock(name="FastAPIRequest")

    def _fake_repo(self, models):
        repo = MagicMock()
        repo.get_all_models = AsyncMock(return_value=models)
        repo._models.apply_model_metadata = AsyncMock()
        return repo

    _BOOL_FIELDS = (
        "supports_images",
        "attachment",
        "reasoning",
        "tool_call",
        "structured_output",
        "temperature",
        "open_weights",
    )
    _VALUE_FIELDS = {
        "status": None,
        "family": None,
        "knowledge": None,
        "release_date": None,
        "context_length": None,
        "max_output_tokens": None,
    }

    def _model(self, name, mappings=()):
        model = MagicMock()
        model.name = name
        model.provider_mappings = mappings
        for field in self._BOOL_FIELDS:
            setattr(model, field, False)
        for field, value in self._VALUE_FIELDS.items():
            setattr(model, field, value)
        return model

    def _mapping(self, provider_model_name):
        mapping = MagicMock()
        mapping.provider_model_name = provider_model_name
        return mapping

    def _payload(self, **entries_by_provider) -> dict:
        return {
            provider: {"models": {model_id: entry for model_id, entry in models.items()}}
            for provider, models in entries_by_provider.items()
        }

    async def _run(self, request_obj, repo, data):
        async def fake_fetch_models_dev_data(client):
            return data

        with (
            patch.object(metadata, "fetch_models_dev_data", fake_fetch_models_dev_data),
            patch.object(metadata, "get_config_repository", return_value=repo),
        ):
            return await sync_model_metadata(request_obj, session=MagicMock())

    async def test_http_status_error_returns_failure(self, request_obj):
        async def boom(client):
            raise httpx2.HTTPStatusError(
                "bad", request=MagicMock(), response=MagicMock(status_code=500)
            )

        with patch.object(metadata, "fetch_models_dev_data", boom):
            response = await sync_model_metadata(request_obj, session=MagicMock())

        assert not response.success
        assert "models.dev" in (response.error or "")

    async def test_request_error_returns_failure(self, request_obj):
        async def boom(client):
            raise httpx2.RequestError("network down")

        with patch.object(metadata, "fetch_models_dev_data", boom):
            response = await sync_model_metadata(request_obj, session=MagicMock())

        assert not response.success
        assert "Network error" in (response.error or "")

    async def test_preview_reports_changes_without_writing(self, request_obj):
        model = self._model("gpt-4o", [self._mapping("gpt-4o")])
        model.reasoning = True  # stored True; catalog says False -> change
        repo = self._fake_repo([model])

        response = await self._run(
            request_obj, repo, self._payload(openai={"gpt-4o": _catalog_entry(reasoning=False)})
        )

        assert response.success
        assert response.total_models == 1
        assert response.changed_count == 1
        assert response.results[0].selected_source == "openai"
        assert "reasoning" in response.results[0].message
        repo._models.apply_model_metadata.assert_not_awaited()

    async def test_unchanged_model_counts_separately(self, request_obj):
        model = self._model("gpt-4o", [self._mapping("gpt-4o")])
        for field in metadata._SYNCED_FIELDS:
            setattr(
                model,
                field,
                {
                    "supports_images": True,
                    "attachment": True,
                    "reasoning": True,
                    "tool_call": False,
                    "structured_output": True,
                    "temperature": True,
                    "open_weights": False,
                    "status": "beta",
                    "family": "gpt",
                    "knowledge": "2025-01-01",
                    "release_date": "2025-02-01",
                    "context_length": 128000,
                    "max_output_tokens": 16384,
                }[field],
            )
        repo = self._fake_repo([model])

        response = await self._run(
            request_obj, repo, self._payload(openai={"gpt-4o": _catalog_entry()})
        )

        assert response.unchanged_count == 1
        assert response.changed_count == 0
        assert response.results[0].message == "Metadata unchanged"

    async def test_no_catalog_entry_is_nodata(self, request_obj):
        model = self._model("unknown-model", [self._mapping("unknown-model")])
        repo = self._fake_repo([model])

        response = await self._run(request_obj, repo, self._payload(openai={}))

        assert response.nodata_count == 1
        assert response.results[0].message == "No models.dev entry found"
        assert response.results[0].available_sources == []

    async def test_candidates_collected_across_mappings_and_deduped(self, request_obj):
        model = self._model("gpt-4o", [self._mapping("gpt-4o"), self._mapping("org/gpt-4o")])
        repo = self._fake_repo([model])

        response = await self._run(
            request_obj,
            repo,
            self._payload(
                openai={"gpt-4o": _catalog_entry()},
                azure={"gpt-4o": _catalog_entry(status="deprecated")},
            ),
        )

        result = response.results[0]
        assert [opt.source for opt in result.available_sources] == ["azure", "openai"]
        assert result.selected_source == "azure"
        assert result.available_sources[0].status == "deprecated"

    async def test_suffix_alias_resolves_catalog_entry(self, request_obj):
        model = self._model("gpt-4o", [self._mapping("org/gpt-4o")])
        repo = self._fake_repo([model])

        response = await self._run(
            request_obj, repo, self._payload(openai={"org/gpt-4o": _catalog_entry()})
        )

        assert response.changed_count + response.unchanged_count == 1
        assert response.results[0].selected_source == "openai"

    async def test_model_without_mappings_is_nodata(self, request_obj):
        model = self._model("lonely", [])
        repo = self._fake_repo([model])

        response = await self._run(
            request_obj, repo, self._payload(openai={"gpt-4o": _catalog_entry()})
        )

        assert response.nodata_count == 1


class TestApplyModelMetadataEndpoint:
    """Behavior of the POST /models/metadata/apply endpoint."""

    @pytest.fixture
    def request_obj(self):
        return MagicMock(name="FastAPIRequest")

    def _repo(self, updated):
        repo = MagicMock()
        repo._models.apply_model_metadata = AsyncMock(return_value=updated)
        return repo

    async def test_applies_only_provided_fields(self, request_obj):
        repo = self._repo(MagicMock())

        with (
            patch.object(metadata, "get_config_repository", return_value=repo),
            patch.object(metadata, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_metadata(
                ApplyMetadataRequest(
                    updates=[MetadataUpdateItem(model_name="gpt-4o", reasoning=True)]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert response.success
        assert response.applied_count == 1
        repo._models.apply_model_metadata.assert_awaited_once_with("gpt-4o", {"reasoning": True})
        reload_mock.assert_awaited_once()

    async def test_partial_update_contract_omitted_fields_untouched(self, request_obj):
        """Explicit null clears; omitted fields never reach the repository."""
        repo = self._repo(MagicMock())

        item = MetadataUpdateItem.model_construct(model_name="gpt-4o", family=None, reasoning=True)
        item.model_fields_set.add("family")
        with (
            patch.object(metadata, "get_config_repository", return_value=repo),
            patch.object(metadata, "commit_and_reload", AsyncMock()),
        ):
            await apply_model_metadata(
                ApplyMetadataRequest(updates=[item]), request_obj, session=MagicMock()
            )

        repo._models.apply_model_metadata.assert_awaited_once_with(
            "gpt-4o", {"family": None, "reasoning": True}
        )

    async def test_empty_updates_fails_item(self, request_obj):
        repo = self._repo(MagicMock())

        with patch.object(metadata, "get_config_repository", return_value=repo):
            response = await apply_model_metadata(
                ApplyMetadataRequest(updates=[MetadataUpdateItem(model_name="gpt-4o")]),
                request_obj,
                session=MagicMock(),
            )

        assert not response.success
        assert response.failed_count == 1
        assert response.results[0].message == "No metadata fields provided"
        repo._models.apply_model_metadata.assert_not_awaited()

    async def test_unknown_model_fails_item(self, request_obj):
        repo = self._repo(None)

        with patch.object(metadata, "get_config_repository", return_value=repo):
            response = await apply_model_metadata(
                ApplyMetadataRequest(
                    updates=[MetadataUpdateItem(model_name="ghost", reasoning=True)]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert not response.success
        assert response.failed_count == 1
        assert response.results[0].message == "Model not found"

    async def test_no_reloads_when_nothing_applied(self, request_obj):
        repo = self._repo(None)

        with (
            patch.object(metadata, "get_config_repository", return_value=repo),
            patch.object(metadata, "commit_and_reload", AsyncMock()) as reload_mock,
        ):
            response = await apply_model_metadata(
                ApplyMetadataRequest(
                    updates=[MetadataUpdateItem(model_name="ghost", reasoning=True)]
                ),
                request_obj,
                session=MagicMock(),
            )

        assert response.applied_count == 0
        reload_mock.assert_not_awaited()


def test_sync_metadata_response_shape():
    """The preview response serializes old/options snapshots for the UI."""
    response = SyncMetadataResponse(
        success=True,
        total_models=1,
        changed_count=1,
        unchanged_count=0,
        nodata_count=0,
    )
    assert response.results == []
