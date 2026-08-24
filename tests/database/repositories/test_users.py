"""Tests for the users repository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.users import UserRepository
from llm_proxy.database.tables import UserRecord


class TestUserRepository:
    """Tests for UserRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repo(self, mock_session):
        """Create a UserRepository instance."""
        return UserRepository(mock_session)

    @pytest.fixture
    def mock_user(self):
        """Create a mock user record with a fixed datetime."""
        user = MagicMock(spec=UserRecord)
        user.id = 1
        user.username = "admin"
        user.password_hash = "$2b$12$" + "x" * 53
        user.role = "admin"
        user.is_active = True
        user.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        return user

    @pytest.mark.asyncio
    async def test_get_by_username_found(self, repo, mock_session, mock_user):
        """Test getting a user by username when it exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_username("admin")

        assert result == mock_user
        # Verify the query uses ilike for case-insensitive lookup
        call_stmt = mock_session.execute.call_args[0][0]
        stmt_str = str(call_stmt)
        assert "lower" in stmt_str.lower() and "like" in stmt_str.lower()

    @pytest.mark.asyncio
    async def test_get_by_username_not_found(self, repo, mock_session):
        """Test getting a user by username when it doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_username("ghost")

        assert result is None

    @pytest.mark.asyncio
    async def test_has_admin_true(self, repo, mock_session):
        """has_admin returns True when at least one active admin exists."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 2
        mock_session.execute = AsyncMock(return_value=mock_result)

        assert await repo.has_admin() is True

        # Verify the query filters by role='admin' AND is_active=True
        call_stmt = mock_session.execute.call_args[0][0]
        stmt_str = str(call_stmt)
        assert "role" in stmt_str
        assert "is_active" in stmt_str

    @pytest.mark.asyncio
    async def test_has_admin_false(self, repo, mock_session):
        """has_admin returns False when no active admin exists."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute = AsyncMock(return_value=mock_result)

        assert await repo.has_admin() is False

    @pytest.mark.asyncio
    async def test_create_user(self, repo, mock_session):
        """Test creating a user with a pre-hashed password."""
        password_hash = "$2b$12$" + "x" * 53
        # Mock get_by_username to return None (no duplicate)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.create_user(
            username="Admin",
            password_hash=password_hash,
            role="viewer",
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()
        # Username should be normalized to lowercase
        assert result.username == "admin"
        assert result.password_hash == password_hash
        assert result.role == "viewer"
        assert result.is_active is True

    @pytest.mark.asyncio
    async def test_create_user_invalid_role(self, repo, mock_session):
        """Creating a user with an invalid role should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            await repo.create_user(
                username="admin",
                password_hash="$2b$12$" + "x" * 53,
                role="superadmin",
            )
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, repo, mock_session, mock_user):
        """Creating a user with an existing username should raise ValueError."""
        # First call returns existing user, second call would be the duplicate check
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="already exists"):
            await repo.create_user(
                username="admin",
                password_hash="$2b$12$" + "x" * 53,
                role="viewer",
            )

    @pytest.mark.asyncio
    async def test_create_initial_admin_success(self, repo, mock_session):
        """create_initial_admin should create the first admin atomically."""
        # No existing admin
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.create_initial_admin(
            username="Admin",
            password_hash="$2b$12$" + "x" * 53,
        )

        assert result.username == "admin"  # normalized
        assert result.role == "admin"
        # Verify FOR UPDATE was used
        call_stmt = mock_session.execute.call_args[0][0]
        stmt_str = str(call_stmt).lower().replace(" ", "")
        assert "forupdate" in stmt_str

    @pytest.mark.asyncio
    async def test_create_initial_admin_already_exists(self, repo, mock_session):
        """create_initial_admin should raise ValueError if admin already exists."""
        mock_result = MagicMock()
        mock_result.first.return_value = MagicMock()  # non-None = admin exists
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="Admin user already exists"):
            await repo.create_initial_admin(
                username="admin",
                password_hash="$2b$12$" + "x" * 53,
            )


class TestUpdateUsername:
    """Tests for UserRepository.update_username."""

    @pytest.fixture
    def mock_session(self):
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repo(self, mock_session):
        return UserRepository(mock_session)

    @pytest.fixture
    def mock_user(self):
        user = MagicMock(spec=UserRecord)
        user.id = 1
        user.username = "admin"
        user.role = "admin"
        user.is_active = True
        user.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        user.token_version = 5
        return user

    @staticmethod
    def _result(value):
        result = MagicMock()
        result.scalar_one_or_none.return_value = value
        return result

    @pytest.mark.asyncio
    async def test_rename_success_normalizes(self, repo, mock_session, mock_user):
        """A rename normalizes the username, persists it, and bumps token_version."""
        # get_by_id -> user; get_by_username(new) -> None (available)
        mock_session.execute = AsyncMock(side_effect=[self._result(mock_user), self._result(None)])

        result = await repo.update_username(1, "  NewName  ")

        assert result is mock_user
        assert mock_user.username == "newname"
        assert mock_user.token_version == 6
        # Once for the rename, once for the token_version bump.
        assert mock_session.flush.await_count == 2
        mock_session.refresh.assert_awaited_once_with(mock_user)

    @pytest.mark.asyncio
    async def test_noop_same_username(self, repo, mock_session, mock_user):
        """Renaming to the current username (after normalization) is a no-op."""
        mock_session.execute = AsyncMock(return_value=self._result(mock_user))

        result = await repo.update_username(1, " ADMIN ")

        assert result is mock_user
        assert mock_user.username == "admin"
        assert mock_user.token_version == 5
        # Only the get_by_id lookup ran; no uniqueness check, no flush.
        assert mock_session.execute.await_count == 1
        mock_session.flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_not_found_returns_none(self, repo, mock_session):
        mock_session.execute = AsyncMock(return_value=self._result(None))

        result = await repo.update_username(99, "newname")

        assert result is None
        mock_session.flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_taken_username_raises(self, repo, mock_session, mock_user):
        """A username held by another user raises ValueError."""
        other = MagicMock(spec=UserRecord)
        other.id = 2
        other.username = "taken"
        mock_session.execute = AsyncMock(side_effect=[self._result(mock_user), self._result(other)])

        with pytest.raises(ValueError, match="already exists"):
            await repo.update_username(1, "taken")

        assert mock_user.username == "admin"
        assert mock_user.token_version == 5
        mock_session.flush.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_integrity_error_raises_value_error(self, repo, mock_session, mock_user):
        """A race hitting the DB unique constraint surfaces as ValueError."""
        mock_session.execute = AsyncMock(side_effect=[self._result(mock_user), self._result(None)])
        mock_session.flush = AsyncMock(side_effect=IntegrityError("dup", {}, None))

        with pytest.raises(ValueError, match="already exists"):
            await repo.update_username(1, "newname")

        assert mock_user.token_version == 5


class TestUserRepositoryGuards:
    """Repository helpers backing the team-lifecycle endpoints."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repo(self, mock_session):
        """Create a UserRepository instance."""
        return UserRepository(mock_session)

    @pytest.mark.asyncio
    async def test_count_active_admins(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session.execute = AsyncMock(return_value=mock_result)

        assert await repo.count_active_admins() == 3

        call_stmt = mock_session.execute.call_args[0][0]
        stmt_str = str(call_stmt)
        assert "role" in stmt_str
        assert "is_active" in stmt_str

    @pytest.mark.asyncio
    async def test_set_role_valid(self, repo, mock_session):
        user = MagicMock()
        user.id = 2
        user.role = "viewer"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.set_role(2, "admin")

        assert result is not None
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_set_role_invalid_rejected(self, repo, mock_session):
        with pytest.raises(ValueError, match="Invalid role"):
            await repo.set_role(2, "superuser")

    @pytest.mark.asyncio
    async def test_set_active_toggles_flag(self, repo, mock_session):
        user = MagicMock()
        user.id = 2
        user.is_active = True
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.set_active(2, False)

        assert result is not None
        assert user.is_active is False

    @pytest.mark.asyncio
    async def test_set_active_missing_user(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        assert await repo.set_active(99, False) is None

    @pytest.mark.asyncio
    async def test_set_must_change_password(self, repo, mock_session):
        user = MagicMock()
        user.id = 2
        user.must_change_password = False
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=mock_result)

        assert await repo.set_must_change_password(2, True) is True
        assert user.must_change_password is True
