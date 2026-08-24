"""Circuit breaker configuration and observation API endpoints."""

import dataclasses

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from llm_proxy.api.dependencies import require_admin_role

router = APIRouter(
    prefix="/circuit-breaker",
    tags=["configuration"],
    dependencies=[Depends(require_admin_role)],
)


class CircuitStateResponse(BaseModel):
    """Response schema for a single circuit state."""

    key: str
    provider: str
    model: str
    index: int
    state: str
    failure_count: int
    last_failure_time: float
    last_state_change: float
    cooldown_seconds: float


class CircuitBreakerListResponse(BaseModel):
    """Response schema for listing all circuit states."""

    enabled: bool
    config: dict
    circuits: list[CircuitStateResponse]


def _parse_provider_key(key: str) -> tuple[str, str, int]:
    """Parse provider key into components.

    Key format: "provider:model:index"
    Note: model can be empty string (represented as '' in the key).
    """
    try:
        parts = key.rsplit(":", 2)
        if len(parts) == 3:
            provider, model, index_str = parts
            return provider, model, int(index_str)
    except ValueError:
        pass
    # Fallback for malformed keys
    return key, "", 0


@router.get("")
async def list_circuit_states(request: Request) -> CircuitBreakerListResponse:
    """List all circuit breaker states.

    Returns the current state of each tracked provider circuit,
    including OPEN/HALF_OPEN/CLOSED status and failure counts.
    """
    store = getattr(request.app.state, "circuit_breaker", None)
    if store is None:
        return CircuitBreakerListResponse(
            enabled=False,
            config={"enabled": False, "failure_threshold": 5, "cooldown_seconds": 60.0},
            circuits=[],
        )

    all_states = store.get_all_states()
    circuits = []

    for key, state in all_states.items():
        provider, model, index = _parse_provider_key(key)
        circuits.append(
            CircuitStateResponse(
                key=key,
                provider=provider,
                model=model,
                index=index,
                state=state["state"],
                failure_count=state["failure_count"],
                last_failure_time=state["last_failure_time"],
                last_state_change=state["last_state_change"],
                cooldown_seconds=state["cooldown_seconds"],
            )
        )

    # Sort by state (OPEN first) then by provider name
    circuits.sort(key=lambda c: (c.state != "OPEN", c.provider))

    return CircuitBreakerListResponse(
        enabled=store.config.enabled,
        config=dataclasses.asdict(store.config),
        circuits=circuits,
    )


@router.post("/reset")
async def reset_all_circuits(request: Request) -> dict:
    """Reset all circuit breakers to CLOSED state."""
    store = getattr(request.app.state, "circuit_breaker", None)
    if store is not None:
        store.reset()
    return {"reset": "all", "count": store.circuit_count if store else 0}


@router.post("/reset/{provider_key:path}")
async def reset_one_circuit(provider_key: str, request: Request) -> dict:
    """Reset a specific circuit breaker by provider key.

    Args:
        provider_key: The provider key in format "provider:model:index"
    """
    store = getattr(request.app.state, "circuit_breaker", None)
    if store is not None:
        store.reset(provider_key)
        return {"reset": provider_key, "success": True}
    return {"reset": provider_key, "success": False, "error": "circuit breaker not initialized"}
