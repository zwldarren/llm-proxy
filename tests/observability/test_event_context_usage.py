"""EventContext's use of the canonical usage record.

The capture object holds one record (ADR-0025): ``update_usage`` folds every
observation with the intake merge rule, so a sparse or repeated frame adds
facts but cannot withdraw one. The two billable dimensions that have a second
route — the provider-reported cost (also a provider-info extra) and the
web-search count (the searches the proxy executed ride the same field) — stay
single fields that only grow within a turn.
"""

from llm_proxy.models.types import Usage
from llm_proxy.observability.event_context import EventContext


def _context() -> EventContext:
    return EventContext(request_id="req-1", trace_id="trace-1", model="m")


class TestUpdateUsage:
    def test_first_observation_is_copied_in(self):
        context = _context()
        usage = Usage(input_tokens=10, output_tokens=5)
        context.update_usage(usage)
        usage.input_tokens = 999  # the caller keeps (and may log) its record
        assert context.usage is not None
        assert context.usage.input_tokens == 10

    def test_a_partial_frame_cannot_withdraw_a_counter(self):
        context = _context()
        context.update_usage(Usage(input_tokens=100, output_tokens=20, cache_read_input_tokens=50))
        # Anthropic's message_delta restates the prompt without the cache
        # counters its message_start carried beside it.
        context.update_usage(Usage(input_tokens=100, output_tokens=25))
        assert context.usage is not None
        assert context.usage.input_tokens == 100
        assert context.usage.output_tokens == 25
        assert context.usage.cache_read_input_tokens == 50

    def test_a_later_observation_is_copied_in_too(self):
        """The first fold copies the record in; every later fold has to as well, since
        the merge adopts the incoming nested details block when the record has none."""
        from llm_proxy.models.types import PromptTokensDetails

        context = _context()
        context.update_usage(Usage(input_tokens=10, output_tokens=1))
        later = Usage(
            input_tokens=5,
            output_tokens=2,
            prompt_tokens_details=PromptTokensDetails(audio_tokens=7),
        )

        context.update_usage(later)

        assert context.usage is not None
        assert context.usage.prompt_tokens_details is not later.prompt_tokens_details
        later.prompt_tokens_details.audio_tokens = 999
        assert context.usage.prompt_tokens_details.audio_tokens == 7

    def test_a_value_that_is_not_a_record_is_ignored(self):
        context = _context()
        context.update_usage(None)
        context.update_usage("not a record")  # type: ignore[arg-type]
        assert context.usage is None

    def test_provider_reported_cost_is_mirrored(self):
        context = _context()
        context.update_usage(Usage(input_tokens=1, provider_reported_cost=0.25))
        assert context.provider_reported_cost == 0.25


class TestWebSearchRequestCount:
    def test_record_count_mirrors_monotonically(self):
        context = _context()
        context.update_usage(Usage(web_search_requests=3))
        context.update_usage(Usage(web_search_requests=1))
        assert context.web_search_requests == 3

    def test_provider_info_cannot_lower_the_count(self):
        context = _context()
        context.update_usage(Usage(web_search_requests=3))
        context.update_provider_info({"server_tool_use": {"web_search_requests": 1}})
        assert context.web_search_requests == 3

    def test_provider_info_can_raise_the_count(self):
        # The searches the proxy executed ride this route, added on top of the
        # provider's own count by the web-search interceptor.
        context = _context()
        context.update_provider_info({"server_tool_use": {"web_search_requests": 2}})
        context.update_provider_info({"server_tool_use": {"web_search_requests": 5}})
        assert context.web_search_requests == 5
