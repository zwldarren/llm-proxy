"""Tests for the routing training-data export endpoint and its row builders."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import require_admin_role
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.routing import router
from llm_proxy.routing.dataset import (
    feedback_export_row,
    jsonl_line,
    judge_shadow_export_row,
    to_jsonl,
    user_prompt_from_body,
)

ROUTING_LOG = {
    "request_body": {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    },
    "log_metadata": {
        "routing": {
            "tier": "MEDIUM",
            "complexity": 0.41,
            "confidence": 0.72,
            "resolved_model": "gpt-routed",
            "judge": {
                "gate": "confidence<0.55",
                "shadow": True,
                "owns_tier": False,
                "tier": "SIMPLE",
                "probabilities": {"SIMPLE": 0.8, "MEDIUM": 0.15, "COMPLEX": 0.05},
                "confidence": 0.8,
                "escalation_probability": 0.1,
                "rubric_version": 3,
            },
        }
    },
}


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[require_admin_role] = lambda: None
    return TestClient(app)


class TestUserPromptFromBody:
    def test_joins_user_turns_only(self):
        assert user_prompt_from_body(ROUTING_LOG["request_body"]) == "What is 2+2?"

    def test_truncates_to_the_judge_budget(self):
        body = {"messages": [{"role": "user", "content": "x" * 20_000}]}
        prompt = user_prompt_from_body(body)
        assert len(prompt) == 8000

    def test_non_dict_body_is_empty(self):
        assert user_prompt_from_body(None) == ""
        assert user_prompt_from_body({"messages": "nope"}) == ""


class TestFeedbackRow:
    def test_row_carries_the_label_and_the_decision(self):
        row = feedback_export_row(
            feedback={
                "request_id": "req-1",
                "signal": "weak",
                "model": "gpt-routed",
                "created_at": 123.0,
            },
            log=ROUTING_LOG,
        )
        assert row is not None
        assert row["signal"] == "weak"
        assert row["prompt"] == "What is 2+2?"
        assert row["tier"] == "MEDIUM"
        assert row["complexity"] == 0.41

    def test_unknown_signal_is_skipped(self):
        row = feedback_export_row(
            feedback={"request_id": "req-1", "signal": "meh", "model": "m", "created_at": 1.0},
            log=ROUTING_LOG,
        )
        assert row is None

    def test_row_without_a_prompt_is_skipped(self):
        row = feedback_export_row(
            feedback={"request_id": "req-1", "signal": "ok", "model": "m", "created_at": 1.0},
            log={"request_body": {}, "log_metadata": {}},
        )
        assert row is None


class TestJudgeShadowRow:
    def test_row_pairs_the_verdict_with_the_ensemble(self):
        judge_meta = ROUTING_LOG["log_metadata"]["routing"]["judge"]
        row = judge_shadow_export_row(
            judge_log={"log_metadata": {"parent_request_id": "req-1", "judge": judge_meta}},
            parent_log=ROUTING_LOG,
            signal="weak",
        )
        assert row is not None
        assert row["ensemble_tier"] == "MEDIUM"
        assert row["judge"]["tier"] == "SIMPLE"
        assert row["judge"]["shadow"] is True
        assert row["judge"]["owns_tier"] is False
        assert row["judge"]["reason"] == "confidence<0.55"
        assert row["signal"] == "weak"

    def test_unknown_signal_becomes_none_not_dropped(self):
        row = judge_shadow_export_row(
            judge_log={"log_metadata": {"parent_request_id": "req-1", "judge": {}}},
            parent_log=ROUTING_LOG,
            signal="bogus",
        )
        assert row is not None
        assert row["signal"] is None


class TestJsonl:
    def test_one_json_object_per_line(self):
        text = to_jsonl([{"a": 1}, {"b": "两"}])
        lines = text.splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": "两"}

    def test_a_single_line_is_one_json_object(self):
        assert jsonl_line({"b": "两"}) == '{"b": "两"}\n'


class TestExportEndpoint:
    def test_feedback_format(self):
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(
            return_value=[
                {
                    "feedback": {
                        "request_id": "req-1",
                        "signal": "ok",
                        "model": "m",
                        "created_at": 1.0,
                    },
                    "log": ROUTING_LOG,
                }
            ]
        )
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset", params={"format": "feedback"})
        assert response.status_code == 200
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert len(rows) == 1
        assert rows[0]["signal"] == "ok"
        repo.get_feedback_dataset_rows.assert_awaited_once()

    def test_judge_shadow_format(self):
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(return_value=[])
        repo.get_judge_dataset_rows = AsyncMock(
            return_value=[
                {
                    "judge_log": {
                        "log_metadata": {
                            "parent_request_id": "req-1",
                            "judge": ROUTING_LOG["log_metadata"]["routing"]["judge"],
                        }
                    },
                    "parent_log": ROUTING_LOG,
                    "signal": None,
                }
            ]
        )
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset", params={"format": "judge_shadow"})
        assert response.status_code == 200
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert len(rows) == 1
        assert rows[0]["judge"]["tier"] == "SIMPLE"

    def test_all_format_returns_both(self):
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(return_value=[])
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset")
        assert response.status_code == 200
        assert response.text == ""
        repo.get_feedback_dataset_rows.assert_awaited_once()
        repo.get_judge_dataset_rows.assert_awaited_once()

    def test_invalid_format_is_rejected(self):
        client = _client()
        response = client.get("/api/routing/export/dataset", params={"format": "dump"})
        assert response.status_code == 422

    def test_rows_without_prompts_are_dropped_silently(self):
        """Pruned logs yield no row rather than an export error."""
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(
            return_value=[
                {
                    "feedback": {
                        "request_id": "r",
                        "signal": "ok",
                        "model": "m",
                        "created_at": 1.0,
                    },
                    "log": None,
                }
            ]
        )
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset", params={"format": "feedback"})
        assert response.status_code == 200
        assert response.text == ""


def _pair(request_id: str, *, signal: str = "ok") -> dict:
    """One repository ``(feedback, log)`` pair."""
    return {
        "feedback": {"request_id": request_id, "signal": signal, "model": "m", "created_at": 1.0},
        "log": ROUTING_LOG,
    }


class TestStreamedExport:
    """The export streams and pages: a 50k-row sweep must not be one allocation."""

    def test_the_body_streams_as_ndjson(self):
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(return_value=[_pair("req-1")])
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset", params={"format": "feedback"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

    def test_rows_are_fetched_in_pages(self, monkeypatch):
        """A short page ends the sweep; a full one asks for the next offset."""
        import llm_proxy.api.routers.routing as routing_router

        monkeypatch.setattr(routing_router, "_PAGE_ROWS", 2)
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(
            side_effect=[[_pair("req-1"), _pair("req-2")], [_pair("req-3")]]
        )
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get(
                "/api/routing/export/dataset", params={"format": "feedback", "limit": 10}
            )
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert [row["request_id"] for row in rows] == ["req-1", "req-2", "req-3"]
        calls = repo.get_feedback_dataset_rows.await_args_list
        assert [call.kwargs["offset"] for call in calls] == [0, 2]
        assert all(call.kwargs["limit"] == 2 for call in calls)

    def test_the_limit_caps_what_a_sweep_fetches(self, monkeypatch):
        import llm_proxy.api.routers.routing as routing_router

        monkeypatch.setattr(routing_router, "_PAGE_ROWS", 2)
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(
            side_effect=[[_pair("req-1"), _pair("req-2")], [_pair("req-3")]]
        )
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get(
                "/api/routing/export/dataset", params={"format": "feedback", "limit": 3}
            )
        assert len(response.text.splitlines()) == 3
        assert repo.get_feedback_dataset_rows.await_count == 2

    def test_both_corpora_page_independently(self, monkeypatch):
        import llm_proxy.api.routers.routing as routing_router

        monkeypatch.setattr(routing_router, "_PAGE_ROWS", 2)
        client = _client()
        repo = MagicMock()
        repo.get_feedback_dataset_rows = AsyncMock(side_effect=[[_pair("req-1")]])
        repo.get_judge_dataset_rows = AsyncMock(return_value=[])
        with patch("llm_proxy.api.routers.routing.LogRepository", return_value=repo):
            response = client.get("/api/routing/export/dataset", params={"limit": 5})
        assert len(response.text.splitlines()) == 1
        # Each corpus starts its own sweep at offset zero.
        assert repo.get_feedback_dataset_rows.await_args.kwargs["offset"] == 0
        assert repo.get_judge_dataset_rows.await_args.kwargs["offset"] == 0
