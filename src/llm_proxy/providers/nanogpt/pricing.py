"""NanoGPT pricing metadata extraction.

NanoGPT responses include an ``x_nanogpt_pricing`` envelope carrying billing
metadata beyond the basic token counts (amount, currency, payment source,
cache/web-search counters, ...). This shared helper preserves the full
envelope plus known sub-fields so streaming and non-streaming paths surface
the same provider-specific billing details.
"""

from typing import Any

# Known sub-fields mirrored out of the pricing envelope for convenient
# provider_info / usage access. The full envelope is also kept under
# ``nanogpt_pricing``.
_NANOGPT_PRICING_SUBFIELDS = (
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cost",
    "amount",
    "currency",
    "error",
    "paymentSource",
    "cacheReadTokens",
    "cacheWriteTokens",
    "webSearchRequests",
    "youtubeTranscripts",
    "scrapedUrls",
)


def extract_nanogpt_pricing(response: dict[str, Any]) -> dict[str, Any] | None:
    """Extract ``x_nanogpt_pricing`` from a NanoGPT response/chunk.

    Returns ``None`` when the envelope is missing or not a dict. Token counts
    are validated (non-negative ints) and the cost is only kept when positive.
    """
    pricing = response.get("x_nanogpt_pricing")
    if not pricing or not isinstance(pricing, dict):
        return None

    result: dict[str, Any] = {}
    input_tokens = pricing.get("inputTokens")
    output_tokens = pricing.get("outputTokens")
    cost = pricing.get("cost")

    if isinstance(input_tokens, int) and input_tokens >= 0:
        result["input_tokens"] = input_tokens
    if isinstance(output_tokens, int) and output_tokens >= 0:
        result["output_tokens"] = output_tokens
    if isinstance(cost, int | float) and cost > 0:
        result["nanogpt_cost"] = cost

    # Preserve the full pricing envelope and any provider-specific sub-fields.
    result["nanogpt_pricing"] = pricing
    for key in _NANOGPT_PRICING_SUBFIELDS:
        if key in pricing:
            result[key] = pricing[key]

    return result if result else None
