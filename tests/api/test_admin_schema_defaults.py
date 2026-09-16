"""Guard against drift between the admin settings schemas and the runtime config.

The Settings UI is driven by the admin schemas (``llm_proxy.api.schemas.admin``):
it renders their fields and treats their defaults as the effective configuration.
The proxy itself runs on the config types (``llm_proxy.config.types``). When the
two disagree, the panel reports a state that is not in effect — and saving an
unrelated field in the same section can persist the wrong default. Response
Keepalive shipped with exactly this bug (schema said off/30s, runtime was on/60s).
"""

from llm_proxy.api.schemas.admin import (
    CircuitBreakerConfigSchema,
    KeepaliveConfig,
    ResilienceConfig,
    SecurityConfig,
)
from llm_proxy.config.types import ProxyAuthConfig
from llm_proxy.config.types.server import (
    CircuitBreakerParams,
    KeepaliveParams,
    SecurityParams,
    ServerParams,
)


def _mismatch(runtime, admin) -> dict[str, tuple[object, object]]:
    """Default values that differ for fields present in both models."""
    runtime_dump = runtime.model_dump()
    admin_dump = admin.model_dump()
    return {
        name: (runtime_dump[name], admin_dump[name])
        for name in admin_dump
        if name in runtime_dump and runtime_dump[name] != admin_dump[name]
    }


def test_security_schema_matches_runtime():
    assert _mismatch(SecurityParams(), SecurityConfig()) == {}


def test_keepalive_schema_matches_runtime():
    assert _mismatch(KeepaliveParams(), KeepaliveConfig()) == {}


def test_circuit_breaker_schema_matches_runtime():
    assert _mismatch(CircuitBreakerParams(), CircuitBreakerConfigSchema()) == {}


def test_resilience_schema_matches_runtime():
    # ServerParams needs an auth section (its validator requires a JWT secret).
    runtime = ServerParams(auth=ProxyAuthConfig(jwt_secret="x" * 32))
    assert _mismatch(runtime, ResilienceConfig()) == {}
