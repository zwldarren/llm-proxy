"""What a request type carries, rendered for an observability backend (ADR-0044)."""

from llm_proxy.observability.content.registry import (
    OBSERVATION_PROFILES,
    ObservationProfile,
    observation_profile,
)

__all__ = ["OBSERVATION_PROFILES", "ObservationProfile", "observation_profile"]
