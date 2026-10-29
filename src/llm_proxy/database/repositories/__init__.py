"""Database repository modules.

This package contains per-entity repository modules for database operations.
"""

from llm_proxy.database.repositories.api_keys import ApiKeyRepository
from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.repositories.config import ConfigRepository
from llm_proxy.database.repositories.config_models import ModelRepository
from llm_proxy.database.repositories.config_providers import ProviderRepository
from llm_proxy.database.repositories.config_server import ServerConfigRepository
from llm_proxy.database.repositories.invitations import InvitationRepository
from llm_proxy.database.repositories.log_repository import (
    LogRepository,
    _build_audit_content_hash_data,
    compute_content_hash,
)
from llm_proxy.database.repositories.oauth_identities import OAuthIdentityRepository
from llm_proxy.database.repositories.passkeys import PasskeyRepository
from llm_proxy.database.repositories.two_factor import TotpRepository
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.repositories.user_sessions import UserSessionRepository
from llm_proxy.database.repositories.users import UserRepository

__all__ = [
    "BaseRepository",
    "ConfigRepository",
    "LogRepository",
    "_build_audit_content_hash_data",
    "compute_content_hash",
    "ApiKeyRepository",
    "InvitationRepository",
    "ProviderRepository",
    "ModelRepository",
    "OAuthIdentityRepository",
    "PasskeyRepository",
    "ServerConfigRepository",
    "UsageRepository",
    "TotpRepository",
    "UserRepository",
    "UserSessionRepository",
]
