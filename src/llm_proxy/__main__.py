"""FastAPI server for LLM Proxy"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from llm_proxy.config import DatabaseConfigManager
from llm_proxy.config.settings import get_settings
from llm_proxy.database import close_db, init_db
from llm_proxy.observability.logger import get_logger

load_dotenv()

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
LOGGER = get_logger(__name__)

# Bind address defaults; overridable via --host/--port CLI flags.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def ensure_frontend_deps() -> bool:
    """Ensure frontend dependencies are installed."""
    if not FRONTEND_DIR.exists():
        LOGGER.warning("Frontend directory not found")
        return False

    if not shutil.which("bun"):
        LOGGER.error(
            "Failed to setup frontend: 'bun' executable not found in PATH. "
            "Please install bun from https://bun.sh/"
        )
        return False

    try:
        if not (FRONTEND_DIR / "node_modules").exists():
            LOGGER.info("Installing frontend dependencies")
            subprocess.run(["bun", "install"], cwd=FRONTEND_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to install frontend dependencies", extra={"error": str(e)})
        return False


def _proxy_header_kwargs() -> dict:
    """Trust X-Forwarded-* headers from the configured trusted proxies.

    Keeps uvicorn's access log client IP consistent with the application's
    own get_client_ip(): behind a reverse proxy (e.g. Traefik/Dokploy or
    Cloudflare) the raw peer is the proxy, so without this the access log
    shows the proxy IP (e.g. 10.0.1.10) instead of the real client.
    """
    return {
        "proxy_headers": True,
        "forwarded_allow_ips": get_settings().security.trusted_proxies,
    }


def build_frontend() -> None:
    """Build the frontend for production."""
    if not ensure_frontend_deps():
        sys.exit(1)

    try:
        LOGGER.info("Building frontend")
        subprocess.run(["bun", "run", "build"], cwd=FRONTEND_DIR, check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Failed to build frontend", extra={"error": str(e)})
        sys.exit(1)


async def main() -> None:
    """Main entry point for the LLM Proxy server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM Proxy Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Database path for configuration (overrides default)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind the server to (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind the server to (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Log level (overrides config)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to this file in addition to console",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--build-frontend",
        action="store_true",
        help="Build frontend before starting server",
    )
    args = parser.parse_args()

    if args.build_frontend:
        build_frontend()

    if args.config:
        os.environ["LLM_PROXY_DB_PATH"] = args.config

    await init_db()

    from llm_proxy.config import ensure_secrets

    await ensure_secrets()

    config_manager = DatabaseConfigManager()
    # Load once to fail fast on invalid database configuration (providers,
    # models, auth) before binding the port.
    await config_manager.load()

    await close_db()

    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    # Single log-level channel: --log-level CLI flag overrides the LOG_LEVEL
    # env var (LoggingSettings.level), which defaults to INFO.
    log_level = args.log_level or get_settings().logging.level

    # Propagate log level to the application's logging system.
    # LLM_PROXY_LOG_LEVEL is an INTERNAL propagation mechanism, not a
    # user-facing config entry: it must be set as an env var BEFORE
    # uvicorn.run() so that worker processes (uvicorn --reload) inherit it —
    # they import the app directly and never go through main().
    os.environ["LLM_PROXY_LOG_LEVEL"] = log_level
    from llm_proxy.observability.logger import get_logging_manager

    get_logging_manager().set_level(log_level)

    if args.log_file:
        # Also propagate via env var so uvicorn --reload workers pick it up.
        os.environ["LLM_PROXY_LOG_FILE"] = args.log_file
        get_logging_manager().enable_file_logging(args.log_file)

    timeout_keep_alive = get_settings().uvicorn.timeout_keepalive

    if args.reload:
        if args.workers > 1:
            print("Warning: --reload and --workers > 1 are incompatible, ignoring --workers")
        import llm_proxy as llm_proxy_module

        assert llm_proxy_module.__file__ is not None
        uvicorn.run(
            "llm_proxy.api:app",
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=True,
            reload_dirs=[os.path.dirname(llm_proxy_module.__file__)],
            timeout_keep_alive=timeout_keep_alive,
            **_proxy_header_kwargs(),
        )
    elif args.workers > 1:
        uvicorn.run(
            "llm_proxy.api:app",
            host=host,
            port=port,
            workers=args.workers,
            log_level=log_level.lower(),
            timeout_keep_alive=timeout_keep_alive,
            **_proxy_header_kwargs(),
        )
    else:
        server = uvicorn.Server(
            uvicorn.Config(
                "llm_proxy.api:app",
                host=host,
                port=port,
                log_level=log_level.lower(),
                reload=False,
                timeout_keep_alive=timeout_keep_alive,
                **_proxy_header_kwargs(),
            )
        )
        await server.serve()


def main_wrapper() -> None:
    """Sync wrapper to run the async main function."""
    asyncio.run(main())


def main_wrapper_dev() -> None:
    """Development wrapper with live frontend HMR."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=9911)
    parser.add_argument("--host", type=str, default="localhost")
    args, extra_args = parser.parse_known_args()

    from llm_proxy.cli.dev_server import start_dev_servers

    start_dev_servers(
        backend_port=args.port,
        backend_host=args.host,
        extra_args=extra_args,
    )


if __name__ == "__main__":
    main_wrapper()
