"""Logging system"""

import contextlib
import logging
import logging.handlers
import os
import sys
import time
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

import orjson
from pydantic import BaseModel, Field


class LogLevel(StrEnum):
    """Log levels with string values."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """Available log formats."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


class LoggingConfig(BaseModel):
    """Configuration for the logging system."""

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.DETAILED
    enable_file_logging: bool = False
    file_path: str | None = None
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup files to keep")
    enable_console_logging: bool = True
    enable_structured_logging: bool = False
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    middleware_stack_logging: bool = False


class StructuredLogger:
    """Structured logger for better observability."""

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        self._context: dict[str, Any] = {}

    def _log_with_context(
        self, level: int, message: str, extra: dict[str, Any] | None = None, **kwargs
    ) -> None:
        """Log with context and additional fields."""
        log_data = {
            "timestamp": time.time(),
            "message": message,
            **self._context,
            **(extra or {}),
            **kwargs,
        }

        self.logger.log(level, log_data)

    def log(self, level: int, message: str, **kwargs) -> None:
        """Log at specified level with context."""
        self._log_with_context(level, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            }:
                log_data[key] = value

        return orjson.dumps(
            log_data, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        ).decode()


class DetailedFormatter(logging.Formatter):
    """Detailed formatter for better debugging."""

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        msg = record.msg
        if isinstance(msg, dict):
            message = msg.pop("message", str(msg))
            timestamp = msg.pop("timestamp", None)
            if timestamp:
                record.created = timestamp
            if msg:
                extra_str = " | " + " ".join(f"{k}={v}" for k, v in msg.items() if v is not None)
                record.msg = message + extra_str
            else:
                record.msg = message
        return super().format(record)


class SimpleFormatter(logging.Formatter):
    """Simple formatter for clean output."""

    def __init__(self):
        super().__init__(fmt="%(levelname)s: %(message)s")


class LoggingManager:
    """Enhanced logging manager with multiple handlers and formatters."""

    def __init__(self, config: LoggingConfig | None = None):
        self.config = config or LoggingConfig()
        # LLM_PROXY_LOG_LEVEL is an internal propagation channel (set by
        # __main__ from LOG_LEVEL/--log-level), not a user-facing config entry.
        # This is essential for uvicorn --reload: the worker process imports the
        # app directly (bypassing main()), so the env var is the only reliable
        # way to propagate the desired log level.
        env_level = os.environ.get("LLM_PROXY_LOG_LEVEL")
        if env_level:
            with contextlib.suppress(ValueError):
                self.config.level = LogLevel(env_level.upper())
        self._loggers: dict[str, StructuredLogger] = {}
        self._handlers: list[logging.Handler] = []
        self._setup_root_logger()
        # Also pick up --log-file from env var (uvicorn --reload workers).
        env_log_file = os.environ.get("LLM_PROXY_LOG_FILE")
        if env_log_file:
            self.enable_file_logging(env_log_file)

    def _create_formatter(self) -> logging.Formatter:
        """Create a formatter based on the current config format."""
        if self.config.format == LogFormat.JSON:
            return JSONFormatter()
        if self.config.format == LogFormat.SIMPLE:
            return SimpleFormatter()
        return DetailedFormatter()

    def _setup_root_logger(self) -> None:
        """Setup the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.value))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Suppress noisy third-party loggers — keep them at WARNING so
        # errors are still visible, but DEBUG/INFO chatter is hidden.
        _NOISY_LOGGERS = [
            # HTTP connection / transport noise
            "httpcore.http11",
            "httpcore2.connection",
            "httpcore2.http11",
            "httpx2",
            "urllib3.connectionpool",
            # Database query noise
            "aiosqlite",
            # Asyncio event-loop debug
            "asyncio",
            # MCP server internals
            "mcp.server.lowlevel.server",
            "mcp.server.streamable_http_manager",
            # Alembic migration internals
            "alembic.runtime.plugins",
            "alembic.runtime.migration",
        ]
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        # Setup console handler
        if self.config.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._create_formatter())
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        # Setup file handler
        if self.config.enable_file_logging and self.config.file_path:
            file_path = Path(self.config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(self._create_formatter())
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)

    def set_level(self, level: str | LogLevel) -> None:
        """Update the log level for all loggers at runtime."""
        if isinstance(level, str):
            level = LogLevel(level.upper())
        self.config.level = level
        py_level = getattr(logging, level.value)
        logging.getLogger().setLevel(py_level)
        for logger in self._loggers.values():
            logger.logger.setLevel(py_level)

    def enable_file_logging(self, file_path: str) -> None:
        """Add a file handler to the root logger at runtime.

        Called from CLI (``--log-file``) to write logs to a file in addition
        to the existing console handler.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=self.config.max_file_size,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(self._create_formatter())
        file_handler.setLevel(getattr(logging, self.config.level.value))
        logging.getLogger().addHandler(file_handler)
        self._handlers.append(file_handler)
        self.config.enable_file_logging = True
        self.config.file_path = file_path

    def get_logger(self, name: str) -> StructuredLogger:
        """Get a structured logger instance."""
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self.config.level)
        return self._loggers[name]


# Module-level global logging manager
_logging_manager: LoggingManager | None = None
_logging_manager_lock = Lock()


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager."""
    global _logging_manager
    if _logging_manager is None:
        with _logging_manager_lock:
            if _logging_manager is None:
                _logging_manager = LoggingManager()
    return _logging_manager


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return get_logging_manager().get_logger(name)
