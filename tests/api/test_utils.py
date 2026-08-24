"""Tests for API utility functions."""

from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from pydantic import BaseModel

from llm_proxy.api.utils import (
    add_info_endpoint,
    create_standard_router,
    create_traced_handler,
)
from llm_proxy.config.settings import SecuritySettings, Settings, set_settings
from llm_proxy.core.request_utils import get_client_ip


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset global settings after each test."""
    from llm_proxy.config.settings import reset_settings as _reset

    yield
    _reset()


def _make_request(*, peer_ip: str, headers: dict[str, str] | None = None) -> MagicMock:
    """Build a minimal FastAPI-style request mock."""
    request = MagicMock(spec=["client", "headers"])
    request.client = MagicMock(host=peer_ip, port=0)
    request.headers = headers or {}
    return request


class TestGetClientIp:
    """Test suite for get_client_ip function."""

    def test_get_client_ip_ignores_x_forwarded_for_without_trusted_proxy(self):
        """X-Forwarded-For is ignored when no trusted proxy is configured."""
        mock_request = _make_request(
            peer_ip="203.0.113.5",
            headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.1"},
        )

        result = get_client_ip(mock_request)

        assert result == "203.0.113.5"

    def test_get_client_ip_ignores_x_real_ip_without_trusted_proxy(self):
        """X-Real-IP is ignored when no trusted proxy is configured."""
        mock_request = _make_request(
            peer_ip="203.0.113.5",
            headers={"X-Real-IP": "192.168.1.2"},
        )

        result = get_client_ip(mock_request)

        assert result == "203.0.113.5"

    def test_get_client_ip_from_direct_client(self):
        """Test extracting IP from direct client connection."""
        mock_request = _make_request(peer_ip="192.168.1.3")

        result = get_client_ip(mock_request)

        assert result == "192.168.1.3"

    def test_get_client_ip_unknown(self):
        """Test returning unknown when no IP can be determined."""
        request = MagicMock(spec=["client", "headers"])
        request.client = None
        request.headers = {}

        result = get_client_ip(request)

        assert result == "unknown"

    def test_get_client_ip_uses_x_forwarded_for_from_trusted_proxy(self):
        """X-Forwarded-For is parsed when the peer is a trusted proxy."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="10.0.0.2",
            headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.1"},
        )

        result = get_client_ip(mock_request)

        assert result == "192.168.1.1"

    def test_get_client_ip_uses_x_real_ip_from_trusted_proxy(self):
        """X-Real-IP is used when the peer is a trusted proxy."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="10.0.0.2",
            headers={"X-Real-IP": "192.168.1.2"},
        )

        result = get_client_ip(mock_request)

        assert result == "192.168.1.2"

    def test_get_client_ip_strips_whitespace(self):
        """Test that IP addresses are properly stripped of whitespace."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="10.0.0.2",
            headers={"X-Forwarded-For": "  192.168.1.1  "},
        )

        result = get_client_ip(mock_request)

        assert result == "192.168.1.1"

    def test_get_client_ip_invalid_forwarded_for_falls_back_to_peer(self):
        """Invalid X-Forwarded-For from a trusted proxy falls back to peer IP."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "127.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="127.0.0.1",
            headers={"X-Forwarded-For": "invalid-ip"},
        )

        result = get_client_ip(mock_request)

        assert result == "127.0.0.1"

    def test_get_client_ip_invalid_forwarded_for_falls_back_to_real_ip(self):
        """Invalid X-Forwarded-For falls back to X-Real-IP when available."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "127.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="127.0.0.1",
            headers={
                "X-Forwarded-For": "invalid-ip",
                "X-Real-IP": "192.168.1.2",
            },
        )

        result = get_client_ip(mock_request)

        assert result == "192.168.1.2"

    def test_get_client_ip_ipv6_forwarded_for_from_trusted_proxy(self):
        """Test extracting IPv6 address from X-Forwarded-From trusted proxy."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "::1/128"})))
        mock_request = _make_request(
            peer_ip="::1",
            headers={"X-Forwarded-For": "2001:db8::1"},
        )

        result = get_client_ip(mock_request)

        assert result == "2001:db8::1"

    def test_get_client_ip_ipv6_real_ip_from_trusted_proxy(self):
        """Test extracting IPv6 address from X-Real-IP from trusted proxy."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "::1/128"})))
        mock_request = _make_request(
            peer_ip="::1",
            headers={"X-Real-IP": "2001:db8::2"},
        )

        result = get_client_ip(mock_request)

        assert result == "2001:db8::2"

    def test_get_client_ip_ignores_headers_from_untrusted_peer(self):
        """When TRUSTED_PROXIES is configured but peer is NOT in the trusted network,
        X-Forwarded-For is still ignored (untrusted source).
        """
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="203.0.113.5",
            headers={"X-Forwarded-For": "1.2.3.4"},
        )

        result = get_client_ip(mock_request)

        assert result == "203.0.113.5"

    def test_get_client_ip_handles_empty_x_forwarded_for(self):
        """Empty X-Forwarded-For from a trusted proxy falls back to peer IP."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="10.0.0.1",
            headers={"X-Forwarded-For": ""},
        )

        result = get_client_ip(mock_request)

        assert result == "10.0.0.1"

    def test_get_client_ip_handles_whitespace_only_entries(self):
        """Whitespace-only entries in X-Forwarded-For are skipped gracefully."""
        set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "10.0.0.0/8"})))
        mock_request = _make_request(
            peer_ip="10.0.0.1",
            headers={"X-Forwarded-For": "1.2.3.4,  , 10.0.0.2"},
        )

        result = get_client_ip(mock_request)

        assert result == "1.2.3.4"


class DummyModel(BaseModel):
    """Dummy model for testing."""

    name: str


class TestCreateTracedHandler:
    """Test suite for create_traced_handler function."""

    @pytest.mark.asyncio
    async def test_create_traced_handler_basic(self):
        """Test creating a traced handler."""

        async def mock_handler(request, fastapi_request):
            return {"received": request.name}

        traced = create_traced_handler("test_handler", mock_handler, DummyModel)

        # Check that the traced handler has correct annotations
        assert "request" in traced.__annotations__
        assert traced.__annotations__["request"] == DummyModel

    @pytest.mark.asyncio
    async def test_traced_handler_calls_original(self):
        """Test that traced handler calls the original handler."""

        async def mock_handler(request, fastapi_request):
            return {"received": request.name}

        traced = create_traced_handler("test_handler", mock_handler, DummyModel)

        mock_request = MagicMock()
        mock_fastapi_request = MagicMock()
        mock_request.name = "test"

        result = await traced(mock_request, mock_fastapi_request)

        assert result == {"received": "test"}


class TestCreateStandardRouter:
    """Test suite for create_standard_router function."""

    @pytest.mark.asyncio
    async def test_create_standard_router_basic(self):
        """Test creating a standard router."""

        async def mock_handler(request, fastapi_request):
            return {"status": "ok"}

        router = create_standard_router(
            request_name="test_endpoint",
            request_model=DummyModel,
            path="/test",
            handler=mock_handler,
        )

        assert isinstance(router, APIRouter)
        assert router.prefix == ""

    def test_create_standard_router_with_prefix(self):
        """Test creating a router with prefix."""

        async def mock_handler(request, fastapi_request):
            return {"status": "ok"}

        router = create_standard_router(
            request_name="test_endpoint",
            request_model=DummyModel,
            path="/test",
            handler=mock_handler,
            prefix="/api/v1",
        )

        assert router.prefix == "/api/v1"

    def test_create_standard_router_with_tags(self):
        """Test creating a router with custom tags."""

        async def mock_handler(request, fastapi_request):
            return {"status": "ok"}

        router = create_standard_router(
            request_name="test_endpoint",
            request_model=DummyModel,
            path="/test",
            handler=mock_handler,
            tags=["custom", "tags"],
        )

        assert router.tags == ["custom", "tags"]

    def test_create_standard_router_default_tags(self):
        """Test that default tags are created from request_name."""

        async def mock_handler(request, fastapi_request):
            return {"status": "ok"}

        router = create_standard_router(
            request_name="my_endpoint",
            request_model=DummyModel,
            path="/test",
            handler=mock_handler,
        )

        assert router.tags == ["my_endpoint"]

    def test_create_standard_router_with_response_model(self):
        """Test creating a router with response model."""

        class ResponseModel(BaseModel):
            status: str

        async def mock_handler(request, fastapi_request):
            return {"status": "ok"}

        router = create_standard_router(
            request_name="test_endpoint",
            request_model=DummyModel,
            path="/test",
            handler=mock_handler,
            response_model=ResponseModel,
        )

        # Router should be created successfully with response model
        assert isinstance(router, APIRouter)


class TestAddInfoEndpoint:
    """Test suite for add_info_endpoint function."""

    @pytest.mark.asyncio
    async def test_add_info_endpoint(self):
        """Test adding an info endpoint to a router."""
        router = APIRouter()

        add_info_endpoint(
            router=router,
            prefix="/info/",
            info_name="version",
            info_data={"version": "1.0.0", "build": "abc123"},
        )

        # The endpoint should be added to the router
        # We can verify by checking the router's routes
        route_paths = [route.path for route in router.routes if isinstance(route, APIRoute)]
        assert "/info/version" in route_paths

    @pytest.mark.asyncio
    async def test_add_info_endpoint_with_tags(self):
        """Test adding an info endpoint with custom tags."""
        router = APIRouter()

        add_info_endpoint(
            router=router,
            prefix="/info/",
            info_name="health",
            info_data={"status": "healthy"},
            info_tags=["monitoring"],
        )

        # Router should have the endpoint
        route_paths = [route.path for route in router.routes if isinstance(route, APIRoute)]
        assert "/info/health" in route_paths

    @pytest.mark.asyncio
    async def test_info_endpoint_returns_data(self):
        """Test that the info endpoint returns the correct data."""
        router = APIRouter()
        info_data = {"version": "1.0.0", "features": ["a", "b"]}

        add_info_endpoint(
            router=router,
            prefix="/info/",
            info_name="version",
            info_data=info_data,
        )

        # Find the route and call its endpoint
        for route in router.routes:
            if isinstance(route, APIRoute) and route.path == "/info/version":
                # Call the endpoint function
                endpoint = route.endpoint
                result = await endpoint()
                assert result == info_data
                break
        else:
            pytest.fail("Info endpoint not found")
