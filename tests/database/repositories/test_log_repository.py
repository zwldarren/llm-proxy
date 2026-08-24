"""Tests for log_repository.py."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.log_repository import (
    LogRepository,
    _build_audit_content_hash_data,
    _schema_cache,
    _SchemaCache,
    compute_chain_hash,
    compute_content_hash,
)
from llm_proxy.database.tables import RequestLog


class TestAuditHashHelpers:
    """Tests for audit log integrity hash helpers."""

    def test_content_hash_includes_bodies(self):
        """Content hash includes request/response headers and bodies."""
        log = MagicMock(spec=RequestLog)
        log.timestamp = 1.0
        log.endpoint = "/api/test"
        log.method = "POST"
        log.status_code = 200
        log.user_identity = "user"
        log.user_id = 42
        log.client_ip = "127.0.0.1"
        log.user_agent = "test-agent"
        log.session_id = "sess-1"
        log.auth_method = "jwt"
        log.api_key_name = None
        log.server_hostname = "host-1"
        log.service_name = "llm-proxy"
        log.event_type = "admin_operation"
        log.action_category = "create"
        log.resource_type = "api_key"
        log.resource_id = "1"
        log.outcome = "success"
        log.error_message = None
        log.model = "gpt-4"
        log.provider = "openai"
        log.cost_usd = 0.01
        log.prompt_tokens = 100
        log.completion_tokens = 50
        log.total_tokens = 150
        log.cache_creation_input_tokens = None
        log.cache_read_input_tokens = None
        log.cached_prompt_tokens = None
        log.cache_savings_usd = None
        log.response_time_ms = 500
        log.ttft_ms = 100
        log.request_headers = {"Authorization": "Bearer secret"}
        log.request_body = {"password": "hunter2"}
        log.response_headers = {}
        log.response_body = {"key": "sk-xxx"}

        original_hash = compute_content_hash(_build_audit_content_hash_data(log))

        # Changing request body should affect the hash
        log.request_body = {"password": "changed"}
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash

    def test_content_hash_includes_cost_and_model(self):
        """Content hash covers cost, model, provider, and token fields."""
        log = MagicMock(spec=RequestLog)
        log.timestamp = 1.0
        log.endpoint = "/api/test"
        log.method = "POST"
        log.status_code = 200
        log.user_identity = "user"
        log.user_id = 42
        log.client_ip = "127.0.0.1"
        log.user_agent = None
        log.session_id = None
        log.auth_method = "jwt"
        log.api_key_name = "key-1"
        log.server_hostname = "host-1"
        log.service_name = "llm-proxy"
        log.event_type = "admin_operation"
        log.action_category = "create"
        log.resource_type = "api_key"
        log.resource_id = "1"
        log.outcome = "success"
        log.error_message = None
        log.model = "gpt-4"
        log.provider = "openai"
        log.cost_usd = 0.01
        log.prompt_tokens = 100
        log.completion_tokens = 50
        log.total_tokens = 150
        log.cache_creation_input_tokens = 10
        log.cache_read_input_tokens = 20
        log.cached_prompt_tokens = 30
        log.cache_savings_usd = 0.005
        log.response_time_ms = 500
        log.ttft_ms = 100
        log.request_headers = {}
        log.request_body = {}
        log.response_headers = {}
        log.response_body = {}

        original_hash = compute_content_hash(_build_audit_content_hash_data(log))

        # Changing cost_usd should affect the hash
        log.cost_usd = 0.02
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash
        log.cost_usd = 0.01  # reset

        # Changing model should affect the hash
        log.model = "gpt-4o"
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash
        log.model = "gpt-4"  # reset

        # Changing user_id should affect the hash
        log.user_id = 99
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash
        log.user_id = 42  # reset

        # Changing provider should affect the hash
        log.provider = "anthropic"
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash
        log.provider = "openai"  # reset

        # Changing total_tokens should affect the hash
        log.total_tokens = 999
        assert compute_content_hash(_build_audit_content_hash_data(log)) != original_hash

    def test_chain_hash_links_to_previous(self):
        """Chain hash incorporates sequence, content hash, and previous chain hash."""
        h1 = compute_chain_hash(1, "content_a", None)
        h2 = compute_chain_hash(2, "content_b", h1)
        h3 = compute_chain_hash(2, "content_b", "different_prev")
        assert h2 != h3
        assert isinstance(h1, str)
        assert isinstance(h2, str)


class TestSchemaCache:
    """Tests for _SchemaCache class."""

    def test_schema_cache_is_instance(self):
        """Test that the module-level _schema_cache is a _SchemaCache instance."""
        assert isinstance(_schema_cache, _SchemaCache)

    def test_set_and_get_table_columns(self):
        """Test setting and getting table columns."""
        cache = _SchemaCache()
        columns = {"id", "timestamp", "request_id"}
        cache.set_table_columns("test_table", columns)
        result = cache.get_table_columns("test_table")
        assert result == columns

    def test_get_table_columns_missing(self):
        """Test getting columns for non-existent table."""
        cache = _SchemaCache()
        result = cache.get_table_columns("nonexistent")
        assert result is None

    def test_set_and_get_tables(self):
        """Test setting and getting table names."""
        cache = _SchemaCache()
        tables = {"table1", "table2", "table3"}
        cache.set_tables(tables)
        result = cache.get_tables()
        assert result == tables

    def test_get_tables_missing(self):
        """Test getting tables when not set."""
        cache = _SchemaCache()
        cache._table_cache = None
        result = cache.get_tables()
        assert result is None

    def test_invalidate(self):
        """Test invalidating cache."""
        cache = _SchemaCache()
        cache.set_table_columns("table1", {"col1"})
        cache.set_tables({"table1"})
        cache.invalidate()
        assert cache.get_table_columns("table1") is None
        assert cache.get_tables() is None

    def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        import threading

        cache = _SchemaCache()
        results = []

        def set_and_get(i):
            cache.set_table_columns(f"table_{i}", {f"col_{i}"})
            results.append(cache.get_table_columns(f"table_{i}"))

        threads = [threading.Thread(target=set_and_get, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == {f"col_{i}"}


class TestLogRepository:
    """Tests for LogRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.bind = MagicMock()
        session.bind.dialect = MagicMock()
        session.bind.dialect.name = "sqlite"
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create a LogRepository instance."""
        return LogRepository(mock_session)

    def test_init(self, mock_session):
        """Test repository initialization."""
        repo = LogRepository(mock_session)
        assert repo.session == mock_session

    @pytest.mark.asyncio
    async def test_create_log(self, repo, mock_session):
        """Test creating a single log entry."""
        log = RequestLog(
            request_id="test-123",
            timestamp=time.time(),
            endpoint="/v1/chat",
            method="POST",
            status_code=200,
        )

        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await repo.create_log(log)

        mock_session.add.assert_called_once_with(log)
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once_with(log)
        assert result == log

    @pytest.mark.asyncio
    async def test_create_logs_bulk_empty(self, repo):
        """Test bulk insert with empty list."""
        result = await repo.create_logs_bulk([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_create_logs_bulk_sqlite(self, repo, mock_session):
        """Test bulk insert with SQLite dialect."""
        logs = [
            RequestLog(
                request_id=f"test-{i}",
                timestamp=time.time(),
                endpoint="/v1/chat",
                method="POST",
                status_code=200,
            )
            for i in range(3)
        ]

        # Mock the schema cache
        with patch.object(
            _schema_cache,
            "get_table_columns",
            return_value={"request_id", "timestamp", "endpoint", "method", "status_code"},
        ):
            mock_session.execute = AsyncMock()
            result = await repo.create_logs_bulk(logs)

        assert result == 3
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_logs_bulk_postgresql(self, repo, mock_session):
        """Test bulk insert with PostgreSQL dialect."""
        mock_session.bind.dialect.name = "postgresql"

        logs = [
            RequestLog(
                request_id=f"test-{i}",
                timestamp=time.time(),
                endpoint="/v1/chat",
                method="POST",
                status_code=200,
            )
            for i in range(3)
        ]

        with patch.object(
            _schema_cache,
            "get_table_columns",
            return_value={"request_id", "timestamp", "endpoint", "method", "status_code"},
        ):
            mock_session.execute = AsyncMock()
            result = await repo.create_logs_bulk(logs)

        assert result == 3

    @pytest.mark.asyncio
    async def test_create_logs_bulk_persists_user_id_for_endpoint_and_web_search(self, repo):
        """Regression: the bulk-insert payload must include user_id.

        Proxy logs (endpoint) and web search logs (web_search) are persisted via
        create_logs_bulk -> _build_insert_values. A prior bug omitted user_id
        from that payload, so non-admin users never saw their own request logs
        (the logs page filters on the numeric user_id column). Audit logs were
        unaffected because they use session.add (ORM insert).
        """
        from llm_proxy.observability.types import LogType

        logs = [
            RequestLog(
                request_id="req-endpoint",
                timestamp=time.time(),
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=200,
                log_type=LogType.ENDPOINT.value,
                user_id=42,
                user_identity="viewer1",
                api_key_name="session:1",
                auth_method="session_api_key",
            ),
            RequestLog(
                request_id="req-websearch",
                timestamp=time.time(),
                endpoint="/v1/chat/completions",
                method="POST",
                status_code=200,
                log_type=LogType.WEB_SEARCH.value,
                user_id=42,
                user_identity="viewer1",
                api_key_name="session:1",
                auth_method="session_api_key",
            ),
        ]

        # Simulate the real schema, which includes the user_id column.
        with patch.object(
            _schema_cache,
            "get_table_columns",
            return_value={
                "request_id",
                "timestamp",
                "endpoint",
                "method",
                "status_code",
                "log_type",
                "user_id",
                "user_identity",
                "api_key_name",
                "auth_method",
            },
        ):
            payloads = await repo._build_filtered_insert_values(logs)

        assert len(payloads) == 2
        for payload in payloads:
            assert payload["user_id"] == 42, (
                f"user_id must be persisted in the bulk-insert payload, got {payload!r}"
            )
            assert payload["user_identity"] == "viewer1"
            assert payload["auth_method"] == "session_api_key"

    @pytest.mark.asyncio
    async def test_delete_old_logs(self, repo, mock_session):
        """Test deleting old logs."""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.delete_old_logs(older_than_ts=time.time() - 86400)

        assert result == 5

    @pytest.mark.asyncio
    async def test_delete_old_logs_with_log_type(self, repo, mock_session):
        """Test deleting old logs with log type filter."""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.delete_old_logs(
            older_than_ts=time.time() - 86400,
            log_type="audit",
        )

        assert result == 3

    def test_normalize_api_log_row(self):
        """Test normalizing API log row."""
        row = {
            "id": 1,
            "request_headers": None,
            "request_body": None,
            "response_headers": None,
            "response_body": None,
            "log_metadata": None,
        }

        result = LogRepository._normalize_api_log_row(row)

        assert result["request_headers"] == {}
        assert result["request_body"] == {}
        assert result["response_headers"] == {}
        assert result["response_body"] == {}
        assert result["log_metadata"] == {}

    def test_normalize_api_log_row_with_values(self):
        """Test normalizing API log row with existing values."""
        row = {
            "id": 1,
            "request_headers": {"Content-Type": "application/json"},
            "request_body": {"model": "gpt-4"},
            "response_headers": {},
            "response_body": {},
            "log_metadata": {},
        }

        result = LogRepository._normalize_api_log_row(row)

        assert result["request_headers"] == {"Content-Type": "application/json"}
        assert result["request_body"] == {"model": "gpt-4"}

    def test_build_insert_values(self):
        """Test building insert values from RequestLog."""
        log = RequestLog(
            request_id="test-123",
            timestamp=1234567890.0,
            endpoint="/v1/chat",
            method="POST",
            status_code=200,
            response_time_ms=100,
            user_identity="user1",
            model="gpt-4",
            provider="openai",
        )

        result = LogRepository._build_insert_values(log)

        assert result["request_id"] == "test-123"
        assert result["timestamp"] == 1234567890.0
        assert result["endpoint"] == "/v1/chat"
        assert result["method"] == "POST"
        assert result["status_code"] == 200

    def test_build_audit_content_hash_data(self):
        """Test building canonical data for content hash."""
        log = RequestLog(
            request_id="test-123",
            timestamp=1234567890.0,
            endpoint="/v1/chat",
            method="POST",
            status_code=200,
            user_identity="user1",
            client_ip="127.0.0.1",
            event_type="chat",
            action_category="create",
            resource_type="completion",
            resource_id="res-1",
            outcome="success",
            request_headers={"Content-Type": "application/json"},
            request_body={"model": "gpt-4"},
            response_headers={"X-Request-Id": "req-1"},
            response_body={"choices": []},
        )

        data = _build_audit_content_hash_data(log)

        assert data["timestamp"] == 1234567890.0
        assert data["endpoint"] == "/v1/chat"
        assert data["request_headers"] == {"Content-Type": "application/json"}
        assert data["request_body"] == {"model": "gpt-4"}
        assert data["response_headers"] == {"X-Request-Id": "req-1"}
        assert data["response_body"] == {"choices": []}

    @pytest.mark.asyncio
    async def test_create_audit_log_with_integrity_no_support(self, repo, mock_session):
        """Test audit log creation when integrity not supported."""
        log = RequestLog(
            request_id="audit-123",
            timestamp=time.time(),
            endpoint="/v1/audit",
            method="POST",
            status_code=200,
            log_type="audit",
        )

        with (
            patch.object(_schema_cache, "get_table_columns", return_value=set()),
            patch.object(_schema_cache, "get_tables", return_value=set()),
        ):
            mock_session.execute = AsyncMock()
            result = await repo.create_audit_log_with_integrity(log)

        assert result == log

    @pytest.mark.asyncio
    async def test_get_log_by_request_id_for_api(self, repo, mock_session):
        """Test getting log for API response."""
        mock_projection_result = MagicMock()
        mock_projection_result.mappings.return_value.one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_projection_result)

        with patch.object(_schema_cache, "get_table_columns", return_value={"id", "request_id"}):
            result = await repo.get_log_by_request_id_for_api("test-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_logs_for_api(self, repo, mock_session):
        """Test getting logs for API response."""
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 0

        mock_select_result = MagicMock()
        mock_select_result.mappings.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_select_result])

        with patch.object(_schema_cache, "get_table_columns", return_value={"id", "timestamp"}):
            result, total = await repo.get_logs_for_api(page=1, page_size=10)

        assert total == 0
        assert result == []

    @pytest.mark.asyncio
    async def test_get_logs_for_api_with_search(self, repo, mock_session):
        """Test searching logs through the unified API response helper."""
        mock_count_result = MagicMock()
        mock_count_result.scalar_one.return_value = 0

        mock_select_result = MagicMock()
        mock_select_result.mappings.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_select_result])

        with patch.object(_schema_cache, "get_table_columns", return_value={"id", "timestamp"}):
            result, total = await repo.get_logs_for_api(page=1, page_size=10, search="test")

        assert total == 0
        assert result == []
