"""Global smart routing configuration (stored in server_config)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RoutingJudgeConfig(BaseModel):
    """System One routing judge settings (ADR-0018).

    The judge is asked to settle decisions the signal ensemble is not sure
    about. It is consulted only when ``enabled`` is set, the turn is the first of
    its conversation, the request's virtual model is listed in ``modes``, and a
    gate predicate fires. Its verdict then owns the tier for that turn; the
    ensemble's floors and caps are applied afterwards either way, so the judge
    can neither lift a request above a cap nor drop it below a floor.

    ``shadow`` keeps the decision unchanged and only records what the judge would
    have said, which is the state to ship first: it is how agreement on live
    traffic is measured before the verdict is trusted. ``shadow_sample_rate``
    additionally consults a small share of turns the gate did *not* fire, the
    only way to see whether the gate itself is letting ambiguous turns through.
    """

    enabled: bool = Field(default=False)
    #: Model to consult, by the name it is registered under in the model list. It
    #: must be marked as a System One model (``supports_systemone``), the same flag
    #: ``/v1/systemone`` requires: a judge call is a System One call, and nothing
    #: else can serve it.
    model: str = Field(default="")
    #: Virtual models whose turns may be judged. ``fast`` is excluded by default:
    #: cheap model choice is that mode's whole point, and a judge call works
    #: against it.
    modes: list[Literal["auto", "fast", "best"]] = ["auto", "best"]
    #: Hard ceiling for the judge call. Overrunning it is an abstention, not a
    #: slow response — the client is waiting, and the added latency budget is
    #: 150 ms at the 95th percentile, so a call that hangs must be abandoned long
    #: before a user would notice the wait.
    deadline_s: float = Field(default=0.5, gt=0.0, le=2.0)

    # ─── Gate: which eligible turns are worth asking about ───
    #: Fire when the ensemble's confidence is below this value.
    confidence_below: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Fire when the ensemble's complexity lands inside the half-open band
    #: ``[low, high)`` — the same convention the public tier mapping uses
    #: (``[0.33, 0.67)`` is MEDIUM). The default band targets the MEDIUM band,
    #: which is where the signals are weakest (ADR-0018).
    complexity_between: tuple[float, float] | None = None

    # ─── Measurement ───
    #: Record the verdict but keep the ensemble's decision.
    shadow: bool = Field(default=True)
    #: Share of *non-gated* eligible turns to consult anyway (0 disables).
    shadow_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    # ─── Judge context budget ───
    #: Prior user turns included, most recent first. The current ask is always
    #: included; assistant and tool content never is (the judge classifies the
    #: request, not the model's own prior output). The shipped gate consults
    #: first turns only, so today the request path never has earlier user turns
    #: to spend this budget on: the excerpt's bounded shape is built regardless,
    #: and the evaluation rig's follow-up cases exercise it, which is what keeps
    #: the budget honest for the day consultation moves past turn one.
    context_turns: int = Field(default=3, ge=0)
    #: Characters allowed per prior user turn.
    context_chars: int = Field(default=4000, gt=0)

    @field_validator("complexity_between")
    @classmethod
    def _band_must_be_ordered(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is None:
            return None
        low, high = value
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                "complexity_between must be a half-open band [low, high) with "
                f"0 <= low < high <= 1, got {list(value)}"
            )
        return (float(low), float(high))

    @property
    def gate_is_open(self) -> bool:
        """True when no predicate is configured, i.e. every eligible turn is judged."""
        return self.confidence_below is None and self.complexity_between is None

    @property
    def is_configured(self) -> bool:
        """True when the judge can actually be consulted."""
        return self.enabled and bool(self.model.strip())


class SmartRoutingConfig(BaseModel):
    enabled: bool = Field(default=False)
    mode_weights: dict[str, float] = Field(
        default_factory=lambda: {"fast": 0.35, "auto": 0.65, "best": 1.0}
    )
    judge: RoutingJudgeConfig = Field(default_factory=RoutingJudgeConfig)

    @staticmethod
    def from_row(value: dict[str, Any] | None) -> SmartRoutingConfig:
        return SmartRoutingConfig(**(value or {}))

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()
