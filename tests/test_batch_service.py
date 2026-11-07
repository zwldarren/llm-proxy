"""Batch service tests over a real SQLite database and faked providers.

The service is where the proxy's own rules live — line validation, one model per
batch, provider resolution, result materialization, ownership — so it is tested
against a migrated database (which also proves the migration applies) rather than
against mocks of its repository.
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import orjson
import pytest

from llm_proxy.batch.service import (
    MAX_BATCH_FILE_BYTES,
    BatchService,
    parse_batch_lines,
)
from llm_proxy.batch.types import BatchResultItem, BatchSnapshot, BatchSubmission
from llm_proxy.batch.wire import batch_object
from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.adapter import get_adapter
from llm_proxy.core.exceptions import NotFoundError, ValidationError
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.tables import UsageRecord, UserRecord

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def db(tmp_path, monkeypatch) -> AsyncIterator[Callable[[], Any]]:
    """A migrated temporary database; returns the session-context factory."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/batch.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-batch-mode-32")

    from llm_proxy.database.connection import get_async_session_context, init_db

    await init_db()
    yield get_async_session_context


class _ConfigManager:
    """The slice of the config manager the batch service reads."""

    def __init__(self, config: Any) -> None:
        self._config = config

    async def get_config(self) -> Any:
        return self._config

    async def get_provider_config(self, provider: str) -> ProviderConfig:
        return self._config.provider_configs[provider]

    async def get_model_config(self, model: str) -> Any:
        return self._config.models.get(model)


class _Services:
    """Stand-in for ``RuntimeServices``: the batch service only reads config."""

    def __init__(self, manager: _ConfigManager) -> None:
        self._manager = manager

    def config_manager(self) -> _ConfigManager:
        return self._manager

    def http_client(self) -> Any:  # pragma: no cover - unused with an injected factory
        raise AssertionError("no HTTP client in these tests")


class FakeBatchAdapter:
    """A minimal adapter implementing the batch capability contract."""

    provider_name = "fake"
    SUPPORTS_BATCH = True
    BATCH_FORWARDS_CHAT_BODY = True

    def __init__(
        self,
        *,
        snapshot: BatchSnapshot | None = None,
        results: list[BatchResultItem] | None = None,
        cancel: BatchSnapshot | None = None,
        supports: bool = True,
        results_error: Exception | None = None,
    ) -> None:
        self.snapshot = snapshot or BatchSnapshot(
            status="validating", request_counts={"total": 0, "completed": 0, "failed": 0}
        )
        self.results = results
        self.cancel = cancel
        self.SUPPORTS_BATCH = supports
        self.results_error = results_error
        self.submitted: dict[str, Any] | None = None
        self.fetch_calls = 0
        self.closed = 0

    async def submit_batch(
        self,
        *,
        model: str,
        requests: list[Any],
        display_name: str,
        completion_window: str = "24h",
    ) -> BatchSubmission:
        self.submitted = {
            "model": model,
            "requests": requests,
            "display_name": display_name,
            "completion_window": completion_window,
        }
        return BatchSubmission(upstream_id="upstream-1", snapshot=self.snapshot)

    async def fetch_batch(self, upstream_id: str) -> BatchSnapshot:
        self.fetch_calls += 1
        return self.snapshot

    async def fetch_batch_results(self, snapshot: BatchSnapshot) -> list[Any] | None:
        if self.results_error is not None:
            raise self.results_error
        return self.results

    async def cancel_batch(self, upstream_id: str) -> BatchSnapshot | None:
        return self.cancel

    def build_batch_request(self, request: Any) -> dict[str, Any]:
        raise AssertionError("the verbatim fake must not translate lines")

    def parse_batch_result(self, payload: dict[str, Any], model: str) -> Any:
        raise AssertionError("the verbatim fake must not translate results")

    async def close(self) -> None:
        self.closed += 1


def _config(
    *,
    model: str = "gpt-5",
    provider: str = "openai-main",
    provider_type: str = "openai",
    upstream_model: str | None = "gpt-5-batch",
    model_name: str | None = None,
) -> Any:
    """The slice of ``ProxyConfig`` the batch service reads.

    A real ``ProxyConfig`` would drag in the auth settings (a required JWT
    secret); the service only touches models, provider configs and the field
    policies, so the fixture supplies exactly those.
    """
    return SimpleNamespace(
        provider_configs={
            provider: ProviderConfig(
                type=provider_type, api_key="sk-test", base_url="https://example.test/v1"
            )
        },
        models={
            model: ModelConfig(
                providers=[
                    ModelProviderConfig(
                        provider=provider, provider_model_name=upstream_model, priority=10
                    )
                ],
                model_name=model_name,
            )
        },
        server_params=SimpleNamespace(unknown_fields_policy=None, unsupported_block_policy=None),
    )


def _service(
    session: Any, config: Any, adapter: Any = None, usage: list[Any] | None = None
) -> BatchService:
    """A service whose upstream is ``adapter`` and whose spend goes to ``usage``.

    Spend is collected rather than written: a real dispatch would start the
    process-lifetime usage writer against the test database. The accounting tests
    pass a list; every other test discards it.
    """
    collected = usage if usage is not None else []

    async def factory(_provider: str) -> Any:
        return adapter

    return BatchService(
        session,
        _Services(_ConfigManager(config)),
        adapter_factory=factory if adapter is not None else None,
        usage_dispatch=collected.append,
    )


def _jsonl(*bodies: dict[str, Any]) -> bytes:
    return b"".join(
        orjson.dumps(
            {
                "custom_id": body.get("custom_id", f"req-{index}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
        + b"\n"
        for index, body in enumerate(bodies)
    )


@pytest.fixture
def response_cls(mock_response_cls):
    """The shared ``MockResponse`` class from ``tests/conftest.py``."""
    return mock_response_cls


@pytest.fixture
def batch_client(make_mock_client):
    """The shared mock-client factory from ``tests/conftest.py``."""
    return make_mock_client


# ── Line validation ───────────────────────────────────────────────────────


class TestLineValidation:
    def test_accepts_the_openai_line_shape(self):
        lines = parse_batch_lines(
            '{"custom_id":"a","method":"POST","url":"/v1/chat/completions",'
            '"body":{"model":"gpt-5","messages":[{"role":"user","content":"hi"}]}}\n'
        )
        assert lines[0].custom_id == "a"
        assert lines[0].model == "gpt-5"

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("", "contains no requests"),
            ("not json\n", "not valid JSON"),
            (
                '{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"m"}}\n{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"m"}}\n',
                "duplicate custom_id",
            ),
            ('{"custom_id":"a","url":"/v1/embeddings","body":{"model":"m"}}\n', "url"),
            (
                '{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"m","stream":true}}\n',
                "stream",
            ),
            ('{"custom_id":"a","url":"/v1/chat/completions","body":{"messages":[]}}\n', "model"),
            (
                '{"custom_id":"a","url":"/v1/chat/completions","body":{"model":"m"}}\n'
                '{"custom_id":"b","url":"/v1/chat/completions","body":{"model":"n"}}\n',
                "same model",
            ),
        ],
    )
    def test_rejects_malformed_lines(self, content, expected):
        with pytest.raises(ValidationError) as excinfo:
            parse_batch_lines(content)
        assert expected in excinfo.value.message


# ── Files ────────────────────────────────────────────────────────────────


class TestFiles:
    async def test_upload_rejects_oversized_and_non_utf8_content(self, db):
        async with db() as session:
            service = _service(session, _config())
            with pytest.raises(ValidationError) as excinfo:
                await service.create_file(
                    owner="key-a",
                    filename="big.jsonl",
                    purpose="batch",
                    content=b"x" * (MAX_BATCH_FILE_BYTES + 1),
                )
            assert excinfo.value.status_code == 413

            with pytest.raises(ValidationError, match="UTF-8"):
                await service.create_file(
                    owner="key-a", filename="bad.jsonl", purpose="batch", content=b"\xff\xfe"
                )

    async def test_upload_rejects_unsupported_purpose(self, db):
        async with db() as session:
            service = _service(session, _config())
            with pytest.raises(ValidationError, match="purpose"):
                await service.create_file(
                    owner="key-a", filename="f.jsonl", purpose="fine-tune", content=b"{}\n"
                )

    async def test_files_are_owned_and_paged(self, db):
        async with db() as session:
            service = _service(session, _config())
            created = [
                await service.create_file(
                    owner="key-a",
                    filename=f"f{index}.jsonl",
                    purpose="batch",
                    content=b'{"custom_id":"a"}\n',
                )
                for index in range(3)
            ]

            with pytest.raises(NotFoundError):
                await service.get_file(owner="key-b", file_id=created[0].id)

            first_page, has_more = await service.list_files(owner="key-a", limit=2)
            assert has_more is True
            assert len(first_page) == 2

            second_page, has_more = await service.list_files(
                owner="key-a", limit=2, after=first_page[-1].id
            )
            assert has_more is False
            # The cursor pages newest-first; the two pages together cover every
            # file exactly once, whatever the id ordering within one second is.
            assert {record.id for record in (*first_page, *second_page)} == {
                record.id for record in created
            }

    async def test_delete_is_owner_scoped(self, db):
        async with db() as session:
            service = _service(session, _config())
            record = await service.create_file(
                owner="key-a", filename="f.jsonl", purpose="batch", content=b"{}\n"
            )
            assert await service.delete_file(owner="key-b", file_id=record.id) is False
            assert await service.delete_file(owner="key-a", file_id=record.id) is True

    async def test_deleting_the_input_of_a_running_batch_is_refused(self, db):
        adapter = FakeBatchAdapter()
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            with pytest.raises(ValidationError) as excinfo:
                await service.delete_file(owner="key-a", file_id=source.id)
            assert excinfo.value.code == "file_in_use"

            # Once the batch is terminal the input file may go.
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
            await service.get_batch(owner="key-a", batch_id=job.id)
            assert await service.delete_file(owner="key-a", file_id=source.id) is True


# ── Batch creation ───────────────────────────────────────────────────────


class TestCreateBatch:
    async def test_rewrites_model_and_persists_the_job(self, db):
        adapter = FakeBatchAdapter(
            snapshot=BatchSnapshot(
                status="validating", request_counts={"total": 2, "completed": 0, "failed": 0}
            )
        )
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl(
                    {"custom_id": "a", "model": "gpt-5", "messages": []},
                    {"custom_id": "b", "model": "gpt-5", "messages": []},
                ),
            )

            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"team": "infra"},
            )

        assert adapter.submitted is not None
        assert adapter.submitted["model"] == "gpt-5-batch"
        assert [item.custom_id for item in adapter.submitted["requests"]] == ["a", "b"]
        assert all(item.body["model"] == "gpt-5-batch" for item in adapter.submitted["requests"])

        body = batch_object(job)
        assert body["object"] == "batch"
        assert body["status"] == "validating"
        assert body["request_counts"] == {"total": 2, "completed": 0, "failed": 0}
        assert body["metadata"] == {"team": "infra"}
        assert body["endpoint"] == "/v1/chat/completions"
        assert body["expires_at"] is None

    async def test_rejects_a_virtual_model(self, db):
        async with db() as session:
            service = _service(session, _config(), FakeBatchAdapter())
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "auto"}),
            )
            with pytest.raises(ValidationError, match="virtual model"):
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )

    async def test_rejects_an_unconfigured_model(self, db):
        async with db() as session:
            service = _service(session, _config(), FakeBatchAdapter())
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "mystery"}),
            )
            with pytest.raises(Exception) as excinfo:
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
            assert getattr(excinfo.value, "status_code", None) == 404

    async def test_names_a_provider_without_batch_support(self, db):
        adapter = FakeBatchAdapter(supports=False)
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            with pytest.raises(ValidationError) as excinfo:
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
        assert "openai-main" in excinfo.value.message
        assert "batch API" in excinfo.value.message

    async def test_a_provider_without_the_capability_attribute_returns_400(self, db):
        """Non-batch adapters do not inherit the mixin's SUPPORTS_BATCH attribute."""

        class NonBatchAdapter:
            provider_name = "plain"

            async def close(self) -> None:
                return None

        async with db() as session:
            service = _service(session, _config(), NonBatchAdapter())
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            with pytest.raises(ValidationError) as excinfo:
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )

        assert excinfo.value.status_code == 400
        assert excinfo.value.code == "batch_unsupported"
        assert "openai-main" in excinfo.value.message

    async def test_enforces_the_key_allowlist_on_the_batch_model(self, db):
        async with db() as session:
            service = _service(session, _config(), FakeBatchAdapter())
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            with pytest.raises(Exception) as excinfo:
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    allowed_models=["another-model"],
                )
        assert getattr(excinfo.value, "status_code", None) == 403
        assert getattr(excinfo.value, "code", None) == "model_not_allowed"

    async def test_rejects_an_unfinished_completion_window(self, db):
        async with db() as session:
            service = _service(session, _config(), FakeBatchAdapter())
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            with pytest.raises(ValidationError, match="completion_window"):
                await service.create_batch(
                    owner="key-a",
                    input_file_id=source.id,
                    endpoint="/v1/chat/completions",
                    completion_window="1h",
                )


# ── Status and results ───────────────────────────────────────────────────


class TestGetBatch:
    async def test_materializes_successes_and_failures_into_two_files(self, db):
        adapter = FakeBatchAdapter(
            results=[
                BatchResultItem(
                    index=0,
                    custom_id="a",
                    body={"id": "chatcmpl-1", "object": "chat.completion"},
                    status_code=200,
                    request_id="req-1",
                ),
                BatchResultItem(
                    index=1,
                    custom_id="b",
                    error={"code": "invalid_request_error", "message": "bad"},
                ),
            ]
        )
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            adapter.snapshot = BatchSnapshot(
                status="completed",
                request_counts={"total": 2, "completed": 1, "failed": 1},
                timestamps={"completed_at": 1_800_000_000},
            )
            refreshed = await service.get_batch(owner="key-a", batch_id=job.id)
            assert refreshed.output_file_id is not None
            assert refreshed.error_file_id is not None

            _, output = await service.get_file_content(
                owner="key-a", file_id=refreshed.output_file_id
            )
            _, errors = await service.get_file_content(
                owner="key-a", file_id=refreshed.error_file_id
            )

        output_line = orjson.loads(output.splitlines()[0])
        assert output_line["custom_id"] == "a"
        assert output_line["response"]["status_code"] == 200
        assert output_line["response"]["body"]["object"] == "chat.completion"
        assert output_line["error"] is None
        # The upstream model name never leaks into the client's result body.
        assert output_line["response"]["body"]["model"] == "gpt-5"

        error_line = orjson.loads(errors.splitlines()[0])
        assert error_line["custom_id"] == "b"
        assert error_line["response"] is None
        assert error_line["error"]["message"] == "bad"

    async def test_results_are_pulled_once(self, db):
        """A second poll must not download (and store) the same results twice."""
        adapter = FakeBatchAdapter(
            results=[BatchResultItem(index=0, custom_id="a", body={"object": "chat.completion"})]
        )
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})

            first = await service.get_batch(owner="key-a", batch_id=job.id)
            second = await service.get_batch(owner="key-a", batch_id=job.id)
            results, _ = await service.list_files(owner="key-a", purpose="batch_output")

        assert first.output_file_id == second.output_file_id
        assert adapter.fetch_calls == 1
        assert len(results) == 1

    async def test_a_completed_batch_with_no_results_is_not_fetched_again(self, db):
        """Zero results is a final answer, not a reason to keep polling upstream."""
        adapter = FakeBatchAdapter()
        adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            refreshed = await service.get_batch(owner="key-a", batch_id=job.id)
            await service.get_batch(owner="key-a", batch_id=job.id)

        assert refreshed.output_file_id is None
        assert refreshed.error_file_id is None
        assert adapter.fetch_calls == 1

    async def test_a_failed_results_pull_is_retried_on_the_next_read(self, db):
        adapter = FakeBatchAdapter(
            results=[BatchResultItem(index=0, custom_id="a", body={"object": "chat.completion"})],
            results_error=RuntimeError("upstream said no"),
        )
        adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            failed = await service.get_batch(owner="key-a", batch_id=job.id)
            failed_file_id = failed.output_file_id
            adapter.results_error = None
            retried = await service.get_batch(owner="key-a", batch_id=job.id)

        assert failed_file_id is None
        assert retried.output_file_id is not None
        assert adapter.fetch_calls == 2

    async def test_terminal_jobs_are_served_from_the_row(self, db):
        adapter = FakeBatchAdapter()
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            adapter.snapshot = BatchSnapshot(status="cancelled", timestamps={})
            await service.get_batch(owner="key-a", batch_id=job.id)
            assert adapter.fetch_calls == 1

            await service.get_batch(owner="key-a", batch_id=job.id)
            assert adapter.fetch_calls == 1

    async def test_unknown_batch_is_not_found(self, db):
        async with db() as session:
            service = _service(session, _config(), FakeBatchAdapter())
            with pytest.raises(NotFoundError):
                await service.get_batch(owner="key-a", batch_id="batch_missing")


# ── Spend ─────────────────────────────────────────────────────────────────

#: One result body carrying the two counters every upstream reports, in the
#: OpenAI shape the materialized output file is rendered in.
_PRICED_BODY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
}


def _priced_config() -> Any:
    """``_config()`` with token prices, so a materialized batch has a real cost."""
    config = _config()
    config.models["gpt-5"] = ModelConfig(
        providers=[
            ModelProviderConfig(
                provider="openai-main", provider_model_name="gpt-5-batch", priority=10
            )
        ],
        model_name="gpt-5",
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
    )
    return config


class TestSpend:
    async def test_a_finished_batch_records_one_usage_row(self, db):
        """One batch is one billable row, priced over every line it reported."""
        adapter = FakeBatchAdapter(
            results=[
                BatchResultItem(index=0, custom_id="a", body=_PRICED_BODY),
                BatchResultItem(index=1, custom_id="b", body=_PRICED_BODY),
            ]
        )
        usage: list[Any] = []
        async with db() as session:
            service = _service(session, _priced_config(), adapter, usage=usage)
            user = UserRecord(username="alice", password_hash="x", role="admin")
            session.add(user)
            await session.flush()
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                user_id=user.id,
            )
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})

            await service.get_batch(owner="key-a", batch_id=job.id)
            # A second poll must not bill the same batch again.
            await service.get_batch(owner="key-a", batch_id=job.id)

        assert len(usage) == 1
        record = usage[0]
        assert record.log_type == "batch"
        assert record.request_id == job.id
        # Stamped with the submission instant, which is what pricing windows use.
        assert record.timestamp == float(job.created_at)
        assert (record.prompt_tokens, record.completion_tokens) == (2000, 1000)
        assert record.total_tokens == 3000
        # 2000 @ $2/1M in + 1000 @ $8/1M out
        assert record.cost_usd == pytest.approx(0.012)
        assert record.model == "gpt-5"
        assert record.provider == "openai-main"
        assert record.api_key_name == "key-a"
        assert record.user_identity is None
        assert record.user_id == user.id
        # Latency is the request path's fact; a batch has none to report.
        assert record.response_time_ms is None
        assert record.ttft_ms is None
        assert record.is_streaming is False

    async def test_a_batch_row_adds_spend_but_not_a_request(self, db):
        """The dashboard span: batch cost counts, a batch never counts as a request."""
        adapter = FakeBatchAdapter(
            results=[BatchResultItem(index=0, custom_id="a", body=_PRICED_BODY)]
        )
        usage: list[Any] = []
        async with db() as session:
            service = _service(session, _priced_config(), adapter, usage=usage)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
            await service.get_batch(owner="key-a", batch_id=job.id)

            # Stand in for the usage writer: the record the service produced is
            # what a spend view then aggregates.
            session.add(UsageRecord(**asdict(usage[0])))
            await session.flush()
            stats = await UsageRepository(session).get_usage_stats(
                log_type=("endpoint", "judge", "batch")
            )
            by_model = await UsageRepository(session).get_usage_by_model(
                log_type=("endpoint", "judge", "batch")
            )

        assert stats["total_requests"] == 0
        assert stats["total_input_tokens"] == 1000
        assert stats["total_cost"] == pytest.approx(0.006)
        assert by_model[0]["requests"] == 0
        assert by_model[0]["cost"] == pytest.approx(0.006)

    async def test_a_console_batch_is_attributed_to_the_user(self, db):
        adapter = FakeBatchAdapter(
            results=[BatchResultItem(index=0, custom_id="a", body=_PRICED_BODY)]
        )
        usage: list[Any] = []
        async with db() as session:
            service = _service(session, _priced_config(), adapter, usage=usage)
            source = await service.create_file(
                owner="user:alice",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="user:alice",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
            await service.get_batch(owner="user:alice", batch_id=job.id)

        assert len(usage) == 1
        assert usage[0].api_key_name is None
        assert usage[0].user_identity == "alice"

    async def test_a_batch_with_no_reported_usage_records_nothing(self, db):
        """No counters means nothing to bill; a $0 row would claim a known cost."""
        adapter = FakeBatchAdapter(
            results=[BatchResultItem(index=0, custom_id="a", body={"object": "chat.completion"})]
        )
        usage: list[Any] = []
        async with db() as session:
            service = _service(session, _priced_config(), adapter, usage=usage)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            adapter.snapshot = BatchSnapshot(status="completed", timestamps={})
            await service.get_batch(owner="key-a", batch_id=job.id)

        assert usage == []


class TestCancel:
    async def test_unsupported_cancel_names_the_provider(self, db):
        adapter = FakeBatchAdapter(cancel=None)
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            with pytest.raises(ValidationError) as excinfo:
                await service.cancel_batch(owner="key-a", batch_id=job.id)
        assert excinfo.value.code == "batch_cancel_unsupported"

    async def test_cancel_applies_the_returned_snapshot(self, db):
        adapter = FakeBatchAdapter(
            cancel=BatchSnapshot(status="cancelling", timestamps={"cancelling_at": 7})
        )
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            cancelled = await service.cancel_batch(owner="key-a", batch_id=job.id)
        assert cancelled.status == "cancelling"
        assert cancelled.cancelling_at == 7

    async def test_a_cancel_that_lands_on_a_final_status_materializes_results(self, db):
        """Gemini can report ``cancelled`` with the partial responses attached."""
        result = BatchResultItem(index=0, custom_id="a", body={"object": "chat.completion"})
        adapter = FakeBatchAdapter(
            results=[result],
            cancel=BatchSnapshot(status="cancelled", timestamps={}, results=[result]),
        )
        async with db() as session:
            service = _service(session, _config(), adapter)
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl({"custom_id": "a", "model": "gpt-5"}),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            cancelled = await service.cancel_batch(owner="key-a", batch_id=job.id)
            _, output = await service.get_file_content(
                owner="key-a", file_id=cancelled.output_file_id
            )

        assert cancelled.status == "cancelled"
        assert cancelled.output_file_id is not None
        assert orjson.loads(output.splitlines()[0])["custom_id"] == "a"


# ── Translated dialect end to end ────────────────────────────────────────


GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "hello from gemini"}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 3,
        "candidatesTokenCount": 4,
        "totalTokenCount": 7,
    },
    "modelVersion": "gemini-3-flash-001",
}


class TestTranslatedDialect:
    async def test_gemini_lines_and_results_are_translated(self, db, response_cls, batch_client):
        client = batch_client(
            [
                response_cls(
                    status_code=200,
                    json_data={
                        "name": "batches/abc",
                        "metadata": {"state": "BATCH_STATE_PENDING"},
                    },
                ),
                response_cls(
                    status_code=200,
                    json_data={
                        "name": "batches/abc",
                        "done": True,
                        "metadata": {
                            "state": "BATCH_STATE_SUCCEEDED",
                            "batchStats": {
                                "requestCount": "1",
                                "successfulRequestCount": "1",
                                "failedRequestCount": "0",
                            },
                        },
                        "response": {
                            "inlinedResponses": {
                                "inlinedResponses": [{"response": GEMINI_RESPONSE}]
                            }
                        },
                    },
                ),
            ]
        )
        adapter = get_adapter("gemini", provider_name="gcp", api_key="test-key")
        adapter._transport._http_client = client
        # The pooled client is shared: the adapter must not close it on exit.
        adapter._transport._owns_http_client = False

        async with db() as session:
            service = _service(
                session,
                _config(
                    model="gemini-flash",
                    provider="gcp",
                    provider_type="gemini",
                    upstream_model="gemini-3-flash",
                ),
                adapter,
            )
            source = await service.create_file(
                owner="key-a",
                filename="in.jsonl",
                purpose="batch",
                content=_jsonl(
                    {
                        "custom_id": "a",
                        "model": "gemini-flash",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
            )
            job = await service.create_batch(
                owner="key-a",
                input_file_id=source.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            # The submitted line reached the adapter as a Gemini request.
            submit_body = client.request.await_args_list[0].kwargs["json"]
            inline = submit_body["batch"]["inputConfig"]["requests"]["requests"][0]
            assert "contents" in inline["request"]
            assert inline["metadata"] == {"custom_id": "a"}

            refreshed = await service.get_batch(owner="key-a", batch_id=job.id)
            _, output = await service.get_file_content(
                owner="key-a", file_id=refreshed.output_file_id
            )

        line = orjson.loads(output.splitlines()[0])
        assert line["custom_id"] == "a"
        body = line["response"]["body"]
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hello from gemini"
        # The client sees the model it asked for, not the upstream's name.
        assert body["model"] == "gemini-flash"
