"""Health check API endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

from llm_proxy.database import get_async_session_context
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health_check(request: Request) -> dict[str, Any]:
    """Get health status of the service."""
    db_healthy = False
    db_status = "unavailable"
    try:
        async with get_async_session_context() as session:
            result = await session.execute(text("SELECT 1"))
            db_healthy = result.scalar() == 1
            db_status = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        logger.warning(f"Health check database error: {e}")

    redis_client = getattr(request.app.state, "redis_client", None)
    redis_status = "disabled"
    redis_healthy = None

    if redis_client and redis_client.is_connected:
        try:
            redis_healthy = await redis_client.health_check()
            redis_status = "healthy" if redis_healthy else "unhealthy"
        except Exception as e:
            logger.warning(f"Health check Redis error: {e}")
            redis_status = "unavailable"
            redis_healthy = False

    overall_healthy = db_healthy and (redis_healthy is None or redis_healthy)

    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "services": {
            "database": {
                "status": db_status,
                "healthy": db_healthy,
            },
            "redis": {
                "status": redis_status,
                "healthy": redis_healthy,
                "enabled": redis_client is not None and redis_client.is_connected,
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/ready")
async def readiness_probe(request: Request) -> dict[str, Any]:
    """Readiness probe for Kubernetes/container orchestration."""
    health = await health_check(request)

    if health["status"] == "healthy":
        return {"status": "ready", "timestamp": health["timestamp"]}

    raise HTTPException(
        status_code=503,
        detail="Service not ready",
    )


@router.get("/live")
async def liveness_probe() -> dict[str, Any]:
    """Liveness probe for Kubernetes/container orchestration."""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}
