from fastapi.testclient import TestClient

from llm_proxy.api import app


class TestSecurityHeaders:
    def test_content_security_policy_header_present(self):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_x_frame_options_denied(self):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.headers.get("X-Frame-Options") == "DENY"
