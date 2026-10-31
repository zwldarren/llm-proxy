"""Validation of the peak/off-peak and service-tier pricing overlays.

Both overlays are authored by hand (models.dev cannot express time-of-day
pricing, and its service-tier data arrives through the sync), so the rules that
make them unambiguous — a readable clock, a known timezone, no overlapping
windows in one timezone, one spelling for each tier name — are enforced here at
every boundary (config load, admin schemas, pricing-apply schema).
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from llm_proxy.config.types.model import (
    ModelConfig,
    ModelProviderConfig,
    PricingTimeWindowConfig,
    minutes_of_day,
    normalize_pricing_time_windows,
    normalize_service_tier_pricing,
)
from llm_proxy.core.exceptions import ValidationError


class TestMinutesOfDay:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("00:00", 0), ("08:30", 510), ("23:59", 1439)],
    )
    def test_valid_clock_times(self, value, expected):
        assert minutes_of_day(value) == expected

    @pytest.mark.parametrize("value", ["24:00", "8:00", "08:60", "", None, 830, "08:00:00"])
    def test_invalid_clock_times(self, value):
        assert minutes_of_day(value) is None


class TestPricingTimeWindowConfig:
    def test_defaults_to_utc_and_every_day(self):
        window = PricingTimeWindowConfig(start="22:00", end="08:00")

        assert window.timezone == "UTC"
        assert window.days == []

    def test_rejects_unreadable_clock(self):
        with pytest.raises(PydanticValidationError, match="HH:MM"):
            PricingTimeWindowConfig(start="late", end="08:00")

    def test_rejects_unknown_timezone(self):
        with pytest.raises(PydanticValidationError, match="timezone"):
            PricingTimeWindowConfig(start="22:00", end="08:00", timezone="Mars/Base")

    def test_rejects_zero_length_window(self):
        with pytest.raises(PydanticValidationError, match="differ"):
            PricingTimeWindowConfig(start="08:00", end="08:00")

    def test_days_are_deduplicated_and_sorted(self):
        window = PricingTimeWindowConfig(start="22:00", end="08:00", days=[4, 1, 4])

        assert window.days == [1, 4]

    def test_rejects_out_of_range_days(self):
        with pytest.raises(PydanticValidationError, match="0 \(Monday\)"):
            PricingTimeWindowConfig(start="22:00", end="08:00", days=[7])


class TestNormalizePricingTimeWindows:
    def _window(self, **overrides) -> PricingTimeWindowConfig:
        defaults = dict(start="22:00", end="08:00")
        defaults.update(overrides)
        return PricingTimeWindowConfig(**defaults)

    def test_falsy_values_pass_through(self):
        assert normalize_pricing_time_windows(None) is None
        assert normalize_pricing_time_windows([]) == []

    def test_sorts_by_timezone_then_start(self):
        windows = normalize_pricing_time_windows(
            [
                self._window(start="20:00", end="21:00"),
                self._window(start="06:00", end="07:00", timezone="Asia/Shanghai"),
                self._window(start="10:00", end="11:00"),
            ]
        )

        assert [window.timezone for window in windows] == ["Asia/Shanghai", "UTC", "UTC"]
        assert [window.start for window in windows] == ["06:00", "10:00", "20:00"]

    def test_rejects_overlapping_windows_in_one_timezone(self):
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [self._window(start="08:00", end="20:00"), self._window(start="19:00", end="23:00")]
            )

    def test_rejects_overlap_across_midnight(self):
        # 22:00–08:00 already covers 23:00; a 23:00–23:30 band would be ambiguous.
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [self._window(start="22:00", end="08:00"), self._window(start="23:00", end="23:30")]
            )

    def test_rejects_overlap_of_the_after_midnight_half(self):
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [self._window(start="22:00", end="08:00"), self._window(start="02:00", end="03:00")]
            )

    def test_allows_disjoint_windows(self):
        windows = normalize_pricing_time_windows(
            [self._window(start="08:00", end="20:00"), self._window(start="20:00", end="08:00")]
        )

        assert len(windows) == 2

    def test_allows_overlap_on_different_days(self):
        windows = normalize_pricing_time_windows(
            [
                self._window(start="08:00", end="20:00", days=[0]),
                self._window(start="08:00", end="20:00", days=[1]),
            ]
        )

        assert len(windows) == 2

    def test_rejects_midnight_wrap_into_a_peer_with_a_disjoint_start_day(self):
        """Sunday 22:00-08:00 covers Monday 01:00, which a Monday band also owns.

        The two windows share no *start* day, so comparing only shared start days
        would let this through even though both price the same instant.
        """
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [
                    self._window(start="00:00", end="02:00", days=[0]),
                    self._window(start="22:00", end="08:00", days=[6]),
                ]
            )

    def test_rejects_the_mirror_midnight_wrap_case(self):
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [
                    self._window(start="22:00", end="08:00", days=[0]),
                    self._window(start="01:00", end="02:00", days=[1]),
                ]
            )

    def test_allows_a_midnight_wrap_that_touches_no_peer(self):
        """A wrap is only a collision when the peer actually covers the tail."""
        windows = normalize_pricing_time_windows(
            [
                self._window(start="00:00", end="02:00", days=[3]),
                self._window(start="22:00", end="08:00", days=[6]),
            ]
        )

        assert len(windows) == 2

    def test_rejects_the_same_clock_in_a_different_timezone(self):
        """08:00-20:00 UTC and 08:00-20:00 Shanghai (UTC+8) share 08:00-12:00 UTC."""
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [
                    self._window(start="08:00", end="20:00", timezone="UTC"),
                    self._window(start="08:00", end="20:00", timezone="Asia/Shanghai"),
                ]
            )

    def test_allows_regional_bands_that_are_disjoint_in_utc(self):
        # Shanghai 20:00-22:00 is 12:00-14:00 UTC, well clear of the UTC band.
        windows = normalize_pricing_time_windows(
            [
                self._window(start="00:00", end="08:00", timezone="UTC"),
                self._window(start="20:00", end="22:00", timezone="Asia/Shanghai"),
            ]
        )

        assert len(windows) == 2

    def test_allows_the_same_nominal_band_across_adjacent_dst_zones(self):
        """London and Berlin share a local clock band but not a real instant.

        London is UTC+0/+1 and Berlin UTC+1/+2, and their offsets always differ;
        testing every offset combination would invent a (UTC+1, UTC+1) pairing
        that no date produces and reject this valid pair.
        """
        windows = normalize_pricing_time_windows(
            [
                self._window(start="09:00", end="10:00", timezone="Europe/London"),
                self._window(start="09:00", end="10:00", timezone="Europe/Berlin"),
            ]
        )

        assert len(windows) == 2

    def test_rejects_a_cross_timezone_midnight_wrap(self):
        """A zone wrapping past midnight can cover the instant another band owns."""
        with pytest.raises(ValidationError, match="must not overlap"):
            normalize_pricing_time_windows(
                [
                    self._window(start="22:00", end="08:00", timezone="Asia/Shanghai"),
                    # 16:00-18:00 UTC is 00:00-02:00 Shanghai, inside the wrap.
                    self._window(start="16:00", end="18:00", timezone="UTC"),
                ]
            )

    def test_unreadable_payload_is_left_for_field_validation(self):
        raw = [{"start": "nope", "end": "08:00"}]

        assert normalize_pricing_time_windows(raw) == raw

    def test_unknown_timezone_is_left_for_field_validation(self):
        raw = [{"start": "22:00", "end": "08:00", "timezone": "Mars/Base"}]

        assert normalize_pricing_time_windows(raw) == raw


class TestNormalizeServiceTierPricing:
    def test_lowercases_keys(self):
        assert list(normalize_service_tier_pricing({"FLEX": {"input_cost_per_1m": 1}})) == ["flex"]

    def test_drops_blank_keys(self):
        assert normalize_service_tier_pricing({"  ": {"input_cost_per_1m": 1}}) == {}

    def test_falsy_values_pass_through(self):
        assert normalize_service_tier_pricing(None) is None
        assert normalize_service_tier_pricing({}) == {}

    def test_provider_config_normalizes_tier_names(self):
        mapping = ModelProviderConfig(
            provider="openai",
            service_tier_pricing={"Priority": {"input_cost_per_1m": 5}},
        )

        assert list(mapping.service_tier_pricing) == ["priority"]

    def test_model_config_normalizes_tier_names(self):
        model = ModelConfig(
            providers=[ModelProviderConfig(provider="openai")],
            service_tier_pricing={"FLEX": {"input_cost_per_1m": 1}},
        )

        assert list(model.service_tier_pricing) == ["flex"]
