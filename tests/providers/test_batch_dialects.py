"""Batch dialect tests: the three upstreams' normalization and HTTP shapes.

Each dialect is exercised twice: through its pure normalization helpers (the
mapping that is easy to get wrong — Gemini strings, OpenRouter's inline results,
OpenAI's error envelope) and through ``submit_batch`` against a mock client, so
the wire shape the proxy actually sends is pinned.
"""

from unittest.mock import MagicMock

import orjson
import pytest

from llm_proxy.batch.types import BatchRequestItem
from llm_proxy.core.adapter import get_adapter
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.providers.gemini.batch import (
    GEMINI_INLINE_LIMIT_BYTES,
    GeminiBatchMixin,
    results_from_operation,
    snapshot_from_operation,
    validate_inline_size,
)
from llm_proxy.providers.openai.batch import (
    batch_jsonl_bytes,
    parse_batch_jsonl,
    result_items_from_lines,
    snapshot_from_batch,
)
from llm_proxy.providers.openrouter.batch import snapshot_from_batch as or_snapshot_from_batch


@pytest.fixture
def response_cls(mock_response_cls):
    """The shared ``MockResponse`` class from ``tests/conftest.py``."""
    return mock_response_cls


@pytest.fixture
def batch_client(make_mock_client):
    """The shared mock-client factory from ``tests/conftest.py``."""
    return make_mock_client


def _adapter(provider: str, client: MagicMock):
    adapter = get_adapter(provider, provider_name=provider, api_key="test-key")
    adapter._transport._http_client = client
    return adapter


# ── OpenAI ────────────────────────────────────────────────────────────────


class TestOpenAIDialect:
    def test_snapshot_maps_status_counts_timestamps_and_errors(self):
        snapshot = snapshot_from_batch(
            {
                "id": "batch_1",
                "status": "completed",
                "request_counts": {"total": 4, "completed": 3, "failed": 1},
                "created_at": 1,
                "in_progress_at": 2,
                "finalizing_at": 3,
                "completed_at": 4,
                "expires_at": 9,
                "output_file_id": "file-out",
                "error_file_id": "file-err",
                "errors": {"object": "list", "data": [{"code": "invalid", "line": 2}]},
            }
        )

        assert snapshot.status == "completed"
        assert snapshot.request_counts == {"total": 4, "completed": 3, "failed": 1}
        assert snapshot.timestamps["completed_at"] == 4
        assert snapshot.timestamps["failed_at"] is None
        assert snapshot.output_file_id == "file-out"
        assert snapshot.error_file_id == "file-err"
        assert snapshot.errors == [{"code": "invalid", "line": 2}]
        assert snapshot.expires_at == 9
        assert snapshot.terminal is True

    def test_snapshot_without_errors_keeps_null(self):
        snapshot = snapshot_from_batch({"id": "batch_1", "status": "in_progress"})
        assert snapshot.errors is None
        assert snapshot.terminal is False

    def test_result_lines_from_output_and_error_files(self):
        output = orjson.dumps(
            {
                "id": "batch_req_1",
                "custom_id": "a",
                "response": {
                    "status_code": 200,
                    "request_id": "req_1",
                    "body": {"id": "chatcmpl-1", "object": "chat.completion"},
                },
                "error": None,
            }
        )
        error = orjson.dumps(
            {
                "id": "batch_req_2",
                "custom_id": "b",
                "response": None,
                "error": {"code": "invalid_request_error", "message": "bad"},
            }
        )

        lines = parse_batch_jsonl(output + b"\n" + error + b"\n")
        items = result_items_from_lines(lines)

        assert [item.custom_id for item in items] == ["a", "b"]
        assert items[0].body == {"id": "chatcmpl-1", "object": "chat.completion"}
        assert items[0].status_code == 200
        assert items[0].request_id == "req_1"
        assert items[1].body is None
        assert items[1].error == {"code": "invalid_request_error", "message": "bad"}

    def test_error_file_items_continue_the_index(self):
        lines = [
            {"custom_id": "c", "response": None, "error": {"message": "bad"}},
        ]
        items = result_items_from_lines(lines, start_index=2)
        assert items[0].index == 2

    @pytest.mark.asyncio
    async def test_submit_uploads_jsonl_then_creates_the_batch(self, response_cls, batch_client):
        client = batch_client(
            [
                response_cls(status_code=200, json_data={"id": "file-abc"}),
                response_cls(
                    status_code=200,
                    json_data={"id": "batch_abc", "status": "validating", "request_counts": {}},
                ),
            ]
        )
        adapter = _adapter("openai", client)

        submission = await adapter.submit_batch(
            model="gpt-5-batch",
            requests=[
                BatchRequestItem(custom_id="a", body={"model": "gpt-5-batch", "messages": []})
            ],
            display_name="batch_1",
        )

        assert submission.upstream_id == "batch_abc"
        assert submission.upstream_file_id == "file-abc"

        upload_kwargs = client.request.await_args_list[0].kwargs
        assert upload_kwargs["data"] == {"purpose": "batch"}
        uploaded = upload_kwargs["files"]["file"][1]
        assert orjson.loads(uploaded.splitlines()[0]) == {
            "custom_id": "a",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-5-batch", "messages": []},
        }
        # The multipart upload must not claim to be a JSON body.
        assert "Content-Type" not in upload_kwargs["headers"]

        create_kwargs = client.request.await_args_list[1].kwargs
        assert create_kwargs["json"] == {
            "input_file_id": "file-abc",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }

    @pytest.mark.asyncio
    async def test_fetch_results_downloads_both_files(self, response_cls, batch_client):
        client = batch_client(
            [
                response_cls(status_code=200, content_data=b'{"custom_id":"a"}\n'),
                response_cls(status_code=200, content_data=b'{"custom_id":"b"}\n'),
            ]
        )
        adapter = _adapter("openai", client)
        snapshot = snapshot_from_batch(
            {
                "id": "batch_abc",
                "status": "completed",
                "output_file_id": "file-out",
                "error_file_id": "file-err",
            }
        )

        items = await adapter.fetch_batch_results(snapshot)

        assert items is not None
        assert [item.custom_id for item in items] == ["a", "b"]
        urls = [call.args[1] for call in client.request.await_args_list]
        assert urls[0].endswith("/files/file-out/content")
        assert urls[1].endswith("/files/file-err/content")


# ── OpenRouter ────────────────────────────────────────────────────────────


class TestOpenRouterDialect:
    def test_snapshot_keeps_inline_results_and_maps_finalized_at(self):
        snapshot = or_snapshot_from_batch(
            {
                "id": "batch_1",
                "status": "completed",
                "finalized_at": 1782097300,
                "request_counts": {"total": 2, "completed": 1, "failed": 1},
                "results": [
                    {
                        "id": "batch_req_1",
                        "custom_id": "a",
                        "response": {
                            "status_code": 200,
                            "request_id": "req_1",
                            "body": {"object": "chat.completion"},
                        },
                        "error": None,
                    },
                    {
                        "id": "batch_req_2",
                        "custom_id": "b",
                        "response": None,
                        "error": {"type": "invalid_request_error", "message": "bad"},
                    },
                ],
            }
        )

        assert snapshot.status == "completed"
        assert snapshot.timestamps["completed_at"] == 1782097300
        assert snapshot.request_counts == {"total": 2, "completed": 1, "failed": 1}
        assert snapshot.results is not None
        assert [item.custom_id for item in snapshot.results] == ["a", "b"]
        assert snapshot.results[0].body == {"object": "chat.completion"}
        assert snapshot.results[1].error == {
            "type": "invalid_request_error",
            "message": "bad",
        }

    def test_failed_batch_error_becomes_an_errors_list(self):
        snapshot = or_snapshot_from_batch(
            {"id": "batch_1", "status": "failed", "error": {"message": "no provider"}}
        )
        assert snapshot.status == "failed"
        assert snapshot.errors == [{"code": "batch_failed", "message": "no provider"}]
        assert snapshot.timestamps["failed_at"] is None

    @pytest.mark.asyncio
    async def test_submit_sends_model_scoped_inline_requests(self, response_cls, batch_client):
        client = batch_client(
            [response_cls(status_code=202, json_data={"id": "batch_1", "status": "validating"})]
        )
        adapter = _adapter("openrouter", client)

        submission = await adapter.submit_batch(
            model="anthropic/claude-haiku-5.5",
            requests=[
                BatchRequestItem(custom_id="a", body={"model": "anthropic/claude-haiku-5.5"})
            ],
            display_name="batch_1",
        )

        assert submission.upstream_id == "batch_1"
        body = client.request.await_args_list[0].kwargs["json"]
        assert body["endpoint"] == "/v1/chat/completions"
        assert body["model"] == "anthropic/claude-haiku-5.5"
        assert body["completion_window"] == "24h"
        assert body["requests"] == [
            {"custom_id": "a", "body": {"model": "anthropic/claude-haiku-5.5"}}
        ]

    @pytest.mark.asyncio
    async def test_cancel_is_unsupported(self, batch_client):
        adapter = _adapter("openrouter", batch_client([]))
        assert await adapter.cancel_batch("batch_1") is None


# ── Gemini ────────────────────────────────────────────────────────────────


class TestGeminiDialect:
    def test_snapshot_maps_batch_state_and_string_counts(self):
        operation = {
            "name": "batches/abc",
            "done": True,
            "metadata": {
                "state": "BATCH_STATE_SUCCEEDED",
                "batchStats": {
                    "requestCount": "2",
                    "successfulRequestCount": "1",
                    "failedRequestCount": "1",
                    "pendingRequestCount": "0",
                },
                "startTime": "2026-11-06T00:00:00Z",
                "endTime": "2026-11-06T01:00:00Z",
            },
            "response": {
                "inlinedResponses": {
                    "inlinedResponses": [
                        {"response": {"candidates": []}, "metadata": {"custom_id": "a"}},
                        {"error": {"code": 3, "message": "bad", "status": "INVALID_ARGUMENT"}},
                    ]
                }
            },
        }

        snapshot = snapshot_from_operation(operation)

        assert snapshot.status == "completed"
        assert snapshot.request_counts == {"total": 2, "completed": 1, "failed": 1}
        assert snapshot.timestamps["in_progress_at"] is not None
        assert snapshot.timestamps["completed_at"] is not None
        assert snapshot.results is not None
        assert snapshot.results[0].custom_id == "a"
        assert snapshot.results[0].body == {"candidates": []}
        assert snapshot.results[1].error == {
            "code": "INVALID_ARGUMENT",
            "message": "bad",
        }

    def test_pending_maps_to_validating_and_running_to_in_progress(self):
        pending = snapshot_from_operation(
            {"name": "batches/a", "metadata": {"state": "BATCH_STATE_PENDING"}}
        )
        running = snapshot_from_operation(
            {"name": "batches/a", "metadata": {"state": "BATCH_STATE_RUNNING"}}
        )
        assert pending.status == "validating"
        assert running.status == "in_progress"
        assert pending.results is None

    @pytest.mark.parametrize(
        ("state", "status"),
        [("BATCH_STATE_CANCELLED", "cancelled"), ("BATCH_STATE_EXPIRED", "expired")],
    )
    def test_other_terminal_states_keep_inline_results(self, state, status):
        """A cancelled/expired operation can still carry the lines completed so far."""
        snapshot = snapshot_from_operation(
            {
                "name": "batches/a",
                "metadata": {"state": state},
                "response": {
                    "inlinedResponses": {
                        "inlinedResponses": [
                            {"response": {"candidates": []}, "metadata": {"custom_id": "a"}}
                        ]
                    }
                },
            }
        )

        assert snapshot.status == status
        assert snapshot.results is not None
        assert snapshot.results[0].custom_id == "a"

    def test_inline_limit_is_strictly_under_decimal_20_mb(self):
        assert GEMINI_INLINE_LIMIT_BYTES == 20_000_000
        validate_inline_size(GEMINI_INLINE_LIMIT_BYTES - 1)
        with pytest.raises(ValidationError) as excinfo:
            validate_inline_size(GEMINI_INLINE_LIMIT_BYTES)
        assert excinfo.value.code == "batch_too_large"

    def test_operation_error_becomes_an_errors_list(self):
        snapshot = snapshot_from_operation(
            {
                "name": "batches/a",
                "done": True,
                "error": {
                    "code": 400,
                    "message": "model not found",
                    "status": "FAILED_PRECONDITION",
                },
                "metadata": {"state": "BATCH_STATE_FAILED"},
            }
        )
        assert snapshot.status == "failed"
        assert snapshot.errors == [{"code": "FAILED_PRECONDITION", "message": "model not found"}]

    def test_results_are_read_from_the_nested_output_envelope(self):
        items = results_from_operation(
            {
                "metadata": {
                    "state": "BATCH_STATE_SUCCEEDED",
                    "output": {
                        "inlinedResponses": {"inlinedResponses": [{"response": {"candidates": []}}]}
                    },
                }
            }
        )
        assert items is not None
        assert items[0].index == 0

    @pytest.mark.asyncio
    async def test_submit_wraps_built_requests_in_the_batch_envelope(
        self, response_cls, batch_client
    ):
        client = batch_client(
            [response_cls(status_code=200, json_data={"name": "batches/abc", "metadata": {}})]
        )
        adapter = _adapter("gemini", client)

        submission = await adapter.submit_batch(
            model="gemini-3-flash",
            requests=[
                BatchRequestItem(custom_id="a", body={"contents": [{"parts": [{"text": "hi"}]}]})
            ],
            display_name="batch_1",
        )

        assert submission.upstream_id == "batches/abc"
        call = client.request.await_args_list[0]
        assert call.args[1].endswith("/models/gemini-3-flash:batchGenerateContent")
        payload = call.kwargs["json"]
        assert payload["batch"]["inputConfig"]["requests"]["requests"] == [
            {
                "request": {"contents": [{"parts": [{"text": "hi"}]}]},
                "metadata": {"custom_id": "a"},
            }
        ]

    @pytest.mark.asyncio
    async def test_cancel_reports_cancelling_while_google_still_runs(
        self, response_cls, batch_client
    ):
        client = batch_client(
            [
                response_cls(status_code=200, json_data={}),
                response_cls(
                    status_code=200,
                    json_data={"name": "batches/abc", "metadata": {"state": "BATCH_STATE_RUNNING"}},
                ),
            ]
        )
        adapter = _adapter("gemini", client)

        snapshot = await adapter.cancel_batch("batches/abc")

        assert snapshot is not None
        assert snapshot.status == "cancelling"
        assert snapshot.timestamps["cancelling_at"] is not None

    @pytest.mark.asyncio
    async def test_batch_is_refused_on_the_interactions_variant(self, batch_client):
        client = batch_client([])
        adapter = get_adapter(
            "gemini",
            provider_name="gemini",
            api_key="k",
            api_variant="interactions",
        )
        adapter._transport._http_client = client

        with pytest.raises(ValidationError) as excinfo:
            await adapter.submit_batch(
                model="gemini-3-flash",
                requests=[BatchRequestItem(custom_id="a", body={})],
                display_name="batch_1",
            )
        assert "generate_content" in excinfo.value.message


# ── Shared helpers ────────────────────────────────────────────────────────


def test_jsonl_helpers_round_trip():
    payload = batch_jsonl_bytes([{"custom_id": "a"}, {"custom_id": "b"}])
    assert payload.endswith(b"\n")
    assert parse_batch_jsonl(payload) == [{"custom_id": "a"}, {"custom_id": "b"}]


def test_gemini_mixin_forwards_no_client_body():
    """The translating side must not forward the OpenAI body verbatim."""
    assert GeminiBatchMixin.BATCH_FORWARDS_CHAT_BODY is False
