"""The xAI adapter's own rule: a materialized continuation stays converted.

Everything declared about the type — base URL, branding, the native Responses
endpoint and which chat-stream side passes through — is asserted once for every
provider of its family in ``tests/providers/test_declared_providers.py``. What
is specific here is the passthrough veto: when the proxy materializes
``previous_response_id`` locally (the request is rewritten to carry the
conversation), the native Responses tier must not be taken, because the
upstream would then answer a different request than the client sent.
"""

from llm_proxy.providers.xai import XAIAdapter
from providers.helpers import make_request, raw_responses

ROUTED_MODEL = "grok-4.6"


def test_materialized_previous_response_is_vetoed():
    adapter = XAIAdapter(api_key="test-key")
    request = make_request(raw_responses(), model=ROUTED_MODEL, protocol_name="openresponses")
    request.previous_response_materialized = True

    assert adapter.supports_native_request("openresponses", request) is False
