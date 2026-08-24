import llm_proxy.protocols.openresponses  # noqa: F401  # ensure serializer registration
from llm_proxy.protocols.registry import get_protocol_serializer

_openresponses_serializer = get_protocol_serializer("openresponses")


class TestOpenResponsesProtocolEndpointErrors:
    def test_invalid_response_type_is_logged(self, caplog):
        """Invalid response type should be logged, not silently returned."""
        with caplog.at_level("ERROR"):
            result = _openresponses_serializer.format_response("invalid_string_response")

        assert any("invalid" in record.message.lower() for record in caplog.records)
        assert result == {"error": "Invalid response type"}
