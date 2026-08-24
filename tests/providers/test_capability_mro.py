"""Regression tests for capability-mixin MRO ordering (ADR-0009).

Adapters compose capabilities as mixins that must precede
``BaseHttpProvider`` in the base-class list; a mixin placed after it is
silently shadowed by ``BaseAdapter``'s ``NotImplementedError`` interface
stubs. ``BaseHttpProvider.__init_subclass__`` rejects the misordering at
class-creation time; these tests pin both the rule itself and every
registered adapter's compliance with it.
"""

import pytest

import llm_proxy.providers  # noqa: F401 — ensure all adapters are registered
from llm_proxy.core.adapter import _ADAPTER_REGISTRY
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.capabilities import (
    AudioCapabilityMixin,
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    ImageCapabilityMixin,
)

#: Capability verbs each mixin implements that ``BaseAdapter`` also declares
#: (as ``NotImplementedError`` stubs or abstract methods) — the shadowing
#: surface the ordering rule protects.
MIXIN_VERBS: dict[type, tuple[str, ...]] = {
    ChatCapabilityMixin: ("stream_chat_completion",),
    EmbeddingCapabilityMixin: ("embeddings",),
    ImageCapabilityMixin: (
        "image_generation",
        "stream_image_generation",
        "image_edit",
        "stream_image_edit",
    ),
    AudioCapabilityMixin: (
        "speech",
        "stream_speech",
        "transcription",
        "stream_transcription",
        "translation",
    ),
}

_BASE_ADAPTER_MODULE = "llm_proxy.core.adapter"

#: Every provider adapter that must be present in the registry. Guards
#: against silent registration loss (a module-level import error aborting
#: provider discovery mid-loop).
EXPECTED_PROVIDERS = {
    "anthropic",
    "chutes",
    "deepseek",
    "gemini",
    "nanogpt",
    "ollama",
    "openai",
    "openai-compatible",
    "openrouter",
}


def test_all_expected_adapters_registered():
    registered = set(_ADAPTER_REGISTRY.get_all())
    missing = EXPECTED_PROVIDERS - registered
    assert not missing, (
        f"adapters missing from the registry: {sorted(missing)} — "
        "an import error during provider discovery likely aborted the loop"
    )


def _registered_adapter_classes() -> set[type]:
    """All distinct adapter classes in the registry."""
    return set(_ADAPTER_REGISTRY.get_all().values())


def test_every_adapter_with_mixins_orders_them_before_the_base():
    """Mixin-provided verbs must never resolve to BaseAdapter's stubs."""
    classes = _registered_adapter_classes()
    assert classes, "adapter registry is empty — providers package not imported?"

    failures = []
    for cls in sorted(classes, key=lambda c: c.__name__):
        for mixin, verbs in MIXIN_VERBS.items():
            if mixin not in cls.__mro__:
                continue
            for verb in verbs:
                if verb not in mixin.__dict__:
                    continue  # the mixin does not provide this verb
                resolved = getattr(cls, verb, None)
                if resolved is not None and resolved.__module__ == _BASE_ADAPTER_MODULE:
                    failures.append(f"{cls.__name__}.{verb}")
    assert not failures, (
        "capability verbs shadowed by BaseAdapter stubs (mixin must precede "
        f"BaseHttpProvider in the base-class list): {failures}"
    )


def test_misordered_mixin_raises_at_class_creation():
    """BaseHttpProvider.__init_subclass__ rejects mixin-after-base ordering."""
    with pytest.raises(TypeError, match="must precede BaseHttpProvider"):

        class MisorderedAdapter(BaseHttpProvider, ChatCapabilityMixin):
            pass


def test_correctly_ordered_mixin_passes():
    """Control: the sanctioned (*CapabilityMixins, BaseHttpProvider) order."""

    class OrderedAdapter(ChatCapabilityMixin, BaseHttpProvider):
        pass

    resolved = OrderedAdapter.stream_chat_completion
    assert resolved is ChatCapabilityMixin.stream_chat_completion
