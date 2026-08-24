"""Tests for list_provider_types() metadata defaults (core/adapter.py)."""

from unittest.mock import patch

from llm_proxy.core.adapter import list_provider_types


class _BareAdapter:
    """Adapter class without branding classvars."""


class _BrandedAdapter:
    """Adapter class with full branding metadata."""

    DISPLAY_NAME_EN = "Test Provider"
    DISPLAY_NAME_ZH = "测试提供商"
    LOBE_ICON_ID = "test"
    LOBE_ICON_VARIANT = "mono"


class _FamilyBase:
    """Family base declaring a mono icon variant."""

    LOBE_ICON_VARIANT = "mono"


class _NoIconChild(_FamilyBase):
    """Subclass inheriting the variant from the family base but declaring no icon."""


def test_metadata_defaults_for_bare_adapters():
    registry = {"bare": _BareAdapter}
    with patch("llm_proxy.core.adapter._ADAPTER_REGISTRY.get_all", return_value=registry):
        types = list_provider_types()
    assert len(types) == 1
    info = types[0]
    assert info.type == "bare"
    assert info.name_en == "bare"
    assert info.name_zh == "bare"
    assert info.icon_id is None
    assert info.icon_variant == "color"


def test_branding_metadata_surface():
    registry = {"branded": _BrandedAdapter}
    with patch("llm_proxy.core.adapter._ADAPTER_REGISTRY.get_all", return_value=registry):
        types = list_provider_types()
    assert len(types) == 1
    info = types[0]
    assert info.name_en == "Test Provider"
    assert info.name_zh == "测试提供商"
    assert info.icon_id == "test"
    assert info.icon_variant == "mono"


def test_variant_normalized_when_icon_missing():
    """A family base's mono variant must not leak onto icon-less subclasses."""
    registry = {"no-icon": _NoIconChild}
    with patch("llm_proxy.core.adapter._ADAPTER_REGISTRY.get_all", return_value=registry):
        types = list_provider_types()
    info = types[0]
    assert info.icon_id is None
    assert info.icon_variant == "color"
