"""Development server management for LLM Proxy.

Manages the lifecycle of frontend and backend dev servers with hot module reload.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def get_frontend_dir() -> Path:
    """Get the frontend directory path."""
    return Path(__file__).parent.parent.parent.parent / "frontend"


def ensure_frontend_deps() -> bool:
    """Ensure frontend dependencies are installed. Returns True if successful."""
    frontend_dir = get_frontend_dir()

    if not frontend_dir.exists():
        logger.warning("Frontend directory not found")
        return False

    import shutil

    if not shutil.which("bun"):
        logger.error(
            "Failed to setup frontend: 'bun' executable not found in PATH. "
            "Please install bun from https://bun.sh/"
        )
        return False

    try:
        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            logger.info("Installing frontend dependencies")
            subprocess.run(["bun", "install"], cwd=frontend_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install frontend dependencies", extra={"error": str(e)})
        return False


def cleanup_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Clean up child processes on exit."""
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def start_dev_servers(
    backend_port: int = 9911,
    backend_host: str = "localhost",
    extra_args: list[str] | None = None,
) -> None:
    """Start both frontend and backend dev servers."""
    if extra_args is None:
        extra_args = []

    frontend_dir = get_frontend_dir()

    if not ensure_frontend_deps():
        sys.exit(1)

    processes: list[subprocess.Popen[bytes]] = []

    def cleanup(_signum: int | None = None, _frame: object = None) -> None:
        cleanup_processes(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        frontend_env = os.environ.copy()
        frontend_env["VITE_API_BASE_URL"] = f"http://{backend_host}:{backend_port}"

        logger.info("Starting frontend dev server at http://0.0.0.0:5173")
        logger.info(f"Frontend will proxy API requests to http://{backend_host}:{backend_port}")
        frontend_proc = subprocess.Popen(
            ["bun", "run", "dev", "--host"],
            cwd=frontend_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=frontend_env,
        )
        processes.append(frontend_proc)

        backend_cmd = [
            sys.executable,
            "-m",
            "llm_proxy",
            "--reload",
            "--port",
            str(backend_port),
            "--host",
            backend_host,
        ]
        backend_cmd.extend(extra_args)

        logger.info(f"Starting backend server at http://{backend_host}:{backend_port}")
        backend_proc = subprocess.Popen(
            backend_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )
        processes.append(backend_proc)

        while True:
            for proc in processes:
                ret = proc.poll()
                if ret is not None:
                    logger.error(f"Process exited with code {ret}")
                    cleanup()
            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        logger.error(f"Error running dev servers: {e}")
        cleanup()
