"""The Moonshot adapter's own guarantee: Kimi models need the reasoning echo.

Everything declared about the type — base URL, branding, the native Anthropic
endpoint — is asserted once for every provider of its family in
``tests/providers/test_declared_providers.py``. What is specific here is the
translated path: ``kimi`` models require the DeepSeek-style reasoning echo on
tool-call turns (the upstream answers HTTP 400 otherwise), which the adapter
opts into through the shared markers.
"""

from llm_proxy.models import InternalRequest
from llm_proxy.providers.moonshot import MoonshotAdapter
from llm_proxy.providers.reasoning import REASONING_ECHO_MODEL_MARKERS
from providers.helpers import make_request, raw_anthropic

ROUTED_MODEL = "kimi-k2.6"


def _request(raw: dict, *, model: str = ROUTED_MODEL) -> InternalRequest:
    return make_request(raw, model=model, protocol_name="anthropic")


def test_kimi_models_require_the_reasoning_echo():
    adapter = MoonshotAdapter(api_key="test-key")

    assert "kimi" in REASONING_ECHO_MODEL_MARKERS
    assert adapter._requires_reasoning_echo(_request(raw_anthropic())) is True
