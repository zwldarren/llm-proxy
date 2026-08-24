"""Log type and audit classification enumerations."""

from enum import StrEnum


class LogType(StrEnum):
    """Log type classification."""

    AUDIT = "audit"
    ENDPOINT = "endpoint"
    MCP = "mcp"
    WEB_SEARCH = "web_search"


class EventType(StrEnum):
    """Audit event types per industry standards.

    Classification of audit events based on OWASP and NIST guidelines.
    """

    AUTHENTICATION = "authentication"  # Login, logout, token refresh
    AUTHORIZATION = "authorization"  # Permission checks
    DATA_ACCESS = "data_access"  # Read operations on sensitive data
    DATA_MODIFICATION = "data_modification"  # Write/update operations
    ADMIN_OPERATION = "admin_operation"  # Admin actions
    SYSTEM_EVENT = "system_event"  # System start/stop, config changes


class ActionCategory(StrEnum):
    """Action categories for audit logs.

    CRUD-style classification of the action performed.
    """

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


class ResourceType(StrEnum):
    """Resource types for audit logs.

    Classification of resources being operated on.
    """

    MODEL = "model"
    API_KEY = "api_key"
    CONFIG = "config"
    LOG = "log"
    PROVIDER = "provider"
    MCP_SERVER = "mcp_server"
    USER = "user"


class Outcome(StrEnum):
    """Outcome of an audit event.

    Result classification based on NIST SP 800-92.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class McpOperationType(StrEnum):
    """MCP operation types for logging."""

    TOOL_CALL = "tool_call"
    TOOL_LIST = "tool_list"
    RESOURCE_READ = "resource_read"
    RESOURCE_LIST = "resource_list"
    PROMPT_GET = "prompt_get"
    PROMPT_LIST = "prompt_list"
    SERVER_START = "server_start"
    SERVER_STOP = "server_stop"


class McpResourceType(StrEnum):
    """MCP resource types for logging."""

    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"
    SERVER = "server"


class WebSearchStatus(StrEnum):
    """Web search status for logging."""

    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAX_USES_EXCEEDED = "max_uses_exceeded"
