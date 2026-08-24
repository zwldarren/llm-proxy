"""Tests for the user sessions repository."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.user_sessions import (
    SESSION_KEY_TTL_HOURS,
    UserSessionRepository,
    generate_session_token,
)
from llm_proxy.database.tables import UserSessionRecord


class TestGenerateSessionToken:
    def test_generates_sk_ui_prefix(self):
        token, token_hash = generate_session_token()
        assert token.startswith("sk-ui-")
        assert len(token) > 20
        assert token[:8] == token[:8]

    def test_hash_is_sha256(self):
        token, token_hash = generate_session_token()
        assert len(token_hash) == 64
        import hashlib

        assert token_hash == hashlib.sha256(token.encode("utf-8")).hexdigest()


class TestUserSessionRepository:
    @pytest.fixture
    def mock_session(self):
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repo(self, mock_session):
        return UserSessionRepository(mock_session)

    @pytest.mark.asyncio
    async def test_create_session(self, repo, mock_session):
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        with patch(
            "llm_proxy.database.repositories.user_sessions.UserSessionRepository.deactivate_user_sessions",
            new=AsyncMock(),
        ):
            record, token = await repo.create_session(user_id=1)

        assert token.startswith("sk-ui-")
        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.user_id == 1
        assert added.is_active is True
        assert added.expires_at > datetime.now(UTC)
        assert added.expires_at < datetime.now(UTC) + timedelta(hours=SESSION_KEY_TTL_HOURS + 1)

    @pytest.mark.asyncio
    async def test_get_session_by_token_found(self, repo, mock_session):
        mock_record = MagicMock(spec=UserSessionRecord)
        mock_record.id = "session-1"
        mock_record.user_id = 1
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_record
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_session_by_token("sk-ui-test123")
        assert result == mock_record

    @pytest.mark.asyncio
    async def test_get_session_by_token_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_session_by_token("sk-ui-nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_deactivate_user_sessions(self, repo, mock_session):
        mock_session.execute = AsyncMock()

        await repo.deactivate_user_sessions(user_id=1)

        call_stmt = mock_session.execute.call_args[0][0]
        stmt_str = str(call_stmt).lower().replace(" ", "")
        assert "update" in stmt_str
        assert "user_sessions" in stmt_str
        assert "is_active" in stmt_str
