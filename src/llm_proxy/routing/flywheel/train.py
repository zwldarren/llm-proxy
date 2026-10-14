"""Train the routing classifier from flywheel labels — and gate the result.

The shipped classifier (``routing/learned.py``) is an averaged perceptron
whose weights were vendored from the external UncommonRoute project; no
training code ever lived in this repo. This module adds it:

* feature construction is byte-identical to the request path — the training
  text is what ``signals/structural.py`` actually classifies (the closed-form
  ask when one is detected, else the last user message, with the compacted
  system prompt), run through ``classifier._extract_all_features``;
* the trainer is the classic multiclass averaged perceptron matching the
  ``model.json`` layout (``weights`` / ``avg_weights`` / ``update_count``);
* evaluation is a seeded split: the best epoch is chosen on a validation slice
  and the gate scores the untouched holdout, so selecting on the gate set
  cannot inflate the result. The artifact is written ONLY when the new model
  beats the shipped one on the holdout (or when no shipped model exists — the
  report says which) — a model that loses stays unwritten. The output goes to
  the sanctioned hot-swap path (``model_online.json``), never to the packaged
  asset;
* the Platt temperature is re-fit from (raw ensemble confidence, correct)
  pairs on the holdout, using the already-shipped grid search.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from llm_proxy.routing import classifier as classifier_module
from llm_proxy.routing.classifier import _extract_all_features
from llm_proxy.routing.flywheel.store import LabelRecord
from llm_proxy.routing.learned import ScriptAgnosticClassifier
from llm_proxy.routing.signal_tuning import (
    compact_structural_system_prompt,
    extract_closed_form_ask,
    strip_client_wrapper_blocks,
)
from llm_proxy.routing.signals.structural import (
    _extract_latest_system_message,
)

TIER_ORDER = ("SIMPLE", "MEDIUM", "COMPLEX")


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """The last user turn's text, empty when there is none."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            parts: list[str] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            return "\n".join(parts)
    return ""


def build_training_features(record: LabelRecord) -> dict[str, float] | None:
    """The exact feature vector the request path would build for this prompt.

    Mirrors ``signals/structural.py::predict``: classification text is the
    detected closed-form ask when present, else the last user message; the
    system prompt is compacted the same way. Returns ``None`` when the record
    has no user text to learn from.
    """
    raw_text = _last_user_text(record.messages)
    # The serving path strips client-injected wrapper scaffolds before the
    # closed-form-ask detection (see ``_extract_last_user_message``); mirror it
    # so a log-sourced record builds the features the deployed model scores.
    stripped = strip_client_wrapper_blocks(raw_text)
    text = (stripped or raw_text).strip()
    if not text:
        return None
    classification_text = extract_closed_form_ask(text) or text
    system_prompt = compact_structural_system_prompt(
        _extract_latest_system_message(record.messages)
    )
    classifier_module._ensure_model_loaded()
    return _extract_all_features(classification_text, system_prompt=system_prompt)


@dataclass
class Example:
    """One training/holdout example: features, label tier, and the messages.

    The messages are kept so the same holdout can be re-read through the full
    ensemble for the Platt re-fit — the ensemble, not the bare classifier, is
    whose confidence the calibration temperature serves.
    """

    features: dict[str, float]
    tier: str
    source: str
    source_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)


def load_examples(records: list[LabelRecord]) -> list[Example]:
    """Labeled examples from records that carry a resolved verdict tier."""
    examples: list[Example] = []
    for record in records:
        if record.verdict_tier not in TIER_ORDER:
            continue
        features = build_training_features(record)
        if features is None:
            continue
        examples.append(
            Example(
                features=features,
                tier=record.verdict_tier,
                source=record.source,
                source_id=record.source_id,
                messages=record.messages,
            )
        )
    return examples


class AveragedPerceptronTrainer:
    """Multiclass averaged perceptron over sparse float features.

    Update rule on a mistake: ``w[target] += f``, ``w[pred] -= f``. The running
    average is tracked over *this training run's* updates (the counter restarts
    at 1), which is what makes warm-starting meaningful: with the vendored
    model's 67k historical updates continued, new updates would be invisible
    in the average the predictor actually scores with.
    """

    def __init__(self, init: ScriptAgnosticClassifier | None = None) -> None:
        tiers = ScriptAgnosticClassifier.TIERS
        if init is not None:
            self._weights = {
                tier: defaultdict(float, init._avg_weights.get(tier, {})) for tier in tiers
            }
        else:
            self._weights = {tier: defaultdict(float) for tier in tiers}
        self._avg_weights = {tier: dict(self._weights[tier]) for tier in tiers}
        self._update_count = 0

    def _scores(self, features: dict[str, float]) -> dict[str, float]:
        return {
            tier: sum(value * weights.get(name, 0.0) for name, value in features.items())
            for tier, weights in self._weights.items()
        }

    def _predict(self, features: dict[str, float]) -> str:
        scores = self._scores(features)
        # Deterministic tie-break toward the *lower* tier: an undecided
        # perceptron should not spend money.
        best = TIER_ORDER[0]
        best_score = scores[best]
        for tier in TIER_ORDER[1:]:
            if scores[tier] > best_score:
                best, best_score = tier, scores[tier]
        return best

    def _apply_update(self, features: dict[str, float], target: str, predicted: str) -> None:
        self._update_count += 1
        denominator = float(self._update_count)
        for tier in TIER_ORDER:
            weights = self._weights[tier]
            delta_sign = 0.0
            if tier == target:
                delta_sign = 1.0
            elif tier == predicted:
                delta_sign = -1.0
            average = self._avg_weights[tier]
            for name, value in features.items():
                if delta_sign:
                    weights[name] += delta_sign * value
                # Running mean of the post-update weights, touched coordinates
                # only; a coordinate's first touch starts the mean at its
                # current weight (there is no earlier state to average over).
                if name in average:
                    average[name] += (weights[name] - average[name]) / denominator
                else:
                    average[name] = weights[name]

    def train_epoch(self, examples: list[Example], rng: random.Random) -> float:
        """One pass; returns the training-set accuracy for the pass."""
        shuffled = list(examples)
        rng.shuffle(shuffled)
        mistakes = 0
        for example in shuffled:
            predicted = self._predict(example.features)
            if predicted != example.tier:
                self._apply_update(example.features, example.tier, predicted)
                mistakes += 1
        return 1.0 - mistakes / max(1, len(shuffled))

    def to_classifier(self, use_ngrams: bool = True) -> ScriptAgnosticClassifier:
        """Export the trained state in the ``model.json`` layout."""
        trained = ScriptAgnosticClassifier(use_ngrams=use_ngrams)
        trained._avg_weights = {
            tier: defaultdict(float, dict(self._avg_weights[tier]))
            for tier in ScriptAgnosticClassifier.TIERS
        }
        trained._weights = {
            tier: defaultdict(float, dict(self._weights[tier]))
            for tier in ScriptAgnosticClassifier.TIERS
        }
        trained._update_count = max(1, self._update_count)
        trained._trained = True
        return trained


def save_classifier(classifier: ScriptAgnosticClassifier, path: Path) -> None:
    """Write the ``model.json`` layout (same keys the loader reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "avg_weights": {
            tier: dict(classifier._avg_weights[tier]) for tier in ScriptAgnosticClassifier.TIERS
        },
        "weights": {
            tier: dict(classifier._weights[tier]) for tier in ScriptAgnosticClassifier.TIERS
        },
        "update_count": classifier._update_count,
        "use_ngrams": classifier._use_ngrams,
    }
    path.write_text(orjson.dumps(payload).decode("utf-8"), encoding="utf-8")


def load_shipped_classifier() -> ScriptAgnosticClassifier | None:
    """The packaged model, for baseline comparison and warm starts."""
    shipped = ScriptAgnosticClassifier()
    path = classifier_module.asset_path("model.json")
    if not path.exists():
        return None
    shipped.load(path)
    return shipped


def predict_with(classifier: ScriptAgnosticClassifier, features: dict[str, float]) -> str:
    _, tier, _ = classifier.predict_complexity(features)
    return tier


@dataclass
class TrainReport:
    """What one training run measured, ready for a human to read."""

    examples: int = 0
    train_size: int = 0
    validation_size: int = 0
    holdout_size: int = 0
    per_epoch_validation: list[float] = field(default_factory=list)
    best_epoch: int = 0
    new_holdout_accuracy: float | None = None
    shipped_holdout_accuracy: float | None = None
    new_holdout_by_tier: dict[str, float] = field(default_factory=dict)
    shipped_holdout_by_tier: dict[str, float] = field(default_factory=dict)
    new_confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    label_source_counts: dict[str, int] = field(default_factory=dict)
    label_verdict_counts: dict[str, int] = field(default_factory=dict)
    platt_temperature_new: float | None = None
    platt_temperature_shipped: float | None = None
    no_baseline: bool = False
    gate_passed: bool = False
    model_written: str | None = None
    calibration_written: str | None = None


def _holdout_accuracy(
    classifier: ScriptAgnosticClassifier, examples: list[Example]
) -> tuple[float, dict[str, float], dict[str, dict[str, int]]]:
    correct = 0
    per_tier_total: dict[str, int] = defaultdict(int)
    per_tier_correct: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = {tier: {} for tier in TIER_ORDER}
    for example in examples:
        predicted = predict_with(classifier, example.features)
        per_tier_total[example.tier] += 1
        confusion[example.tier][predicted] = confusion[example.tier].get(predicted, 0) + 1
        if predicted == example.tier:
            correct += 1
            per_tier_correct[example.tier] += 1
    accuracy = correct / max(1, len(examples))
    by_tier = {
        tier: (per_tier_correct[tier] / per_tier_total[tier]) if per_tier_total[tier] else 0.0
        for tier in TIER_ORDER
    }
    return accuracy, by_tier, confusion


def _platt_pairs(
    examples: list[Example],
    classifier: ScriptAgnosticClassifier,
) -> list[dict[str, Any]]:
    """(raw ensemble confidence, correct) pairs with *classifier* installed.

    The ensemble's raw confidence is read through ``api.classify`` with the
    process-local calibrator swapped for the identity, and Signal B runs the
    given classifier — that is the (confidence, correctness) distribution the
    re-fitted temperature must serve.
    """
    import llm_proxy.routing.api as routing_api
    from llm_proxy.routing.decision.calibration import PlattCalibrator

    saved_classifier = classifier_module._model
    saved_load_flag = classifier_module._model_load_attempted
    saved_calibrator = routing_api._calibrator
    classifier_module._model = classifier
    classifier_module._model_load_attempted = True
    routing_api._calibrator = PlattCalibrator(temperature=1.0)
    pairs: list[dict[str, Any]] = []
    try:
        for example in examples:
            # Re-read the same messages through the ensemble; the training
            # features came from these messages, so the read is consistent by
            # construction.
            result = routing_api.classify(example.messages)
            # The v2 ensemble scores four tiers; both upper bands collapse to
            # COMPLEX in the three-tier label space (see label.py).
            tier = {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX", 3: "COMPLEX"}.get(result.v2.tier_id)
            pairs.append(
                {
                    "confidence": result.v2.confidence,
                    "correct": tier == example.tier,
                }
            )
    finally:
        classifier_module._model = saved_classifier
        classifier_module._model_load_attempted = saved_load_flag
        routing_api._calibrator = saved_calibrator
    return pairs


def train_and_gate(
    records: list[LabelRecord],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 20261014,
    epochs: int = 15,
    warm_start: bool = True,
    out_model: Path | None = None,
    out_calibration: Path | None = None,
) -> TrainReport:
    """Train on the labeled records, evaluate, and gate the artifact write."""
    report = TrainReport()
    all_examples = load_examples(records)
    report.examples = len(all_examples)
    verdict_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for record in records:
        verdict_counts[record.verdict_tier or "abstain"] += 1
        source_counts[record.source] += 1
    report.label_verdict_counts = dict(verdict_counts)
    report.label_source_counts = dict(source_counts)
    if len(all_examples) < 40:
        return report  # too little to split meaningfully; report and stop

    examples = list(all_examples)
    rng = random.Random(seed)
    rng.shuffle(examples)
    holdout_size = max(10, int(len(examples) * holdout_fraction))
    holdout, train_pool = examples[:holdout_size], examples[holdout_size:]
    # Carve a validation slice out of the training pool. The best epoch is
    # chosen on it and the gate then scores the untouched holdout: taking the
    # argmax of ``epochs`` holdout evaluations and gating on the same rows
    # would bias the new model's accuracy upward against a baseline scored
    # once, and could install an overfit-to-holdout model.
    validation_size = min(max(5, int(len(train_pool) * 0.2)), len(train_pool) // 2)
    validation, train = train_pool[:validation_size], train_pool[validation_size:]
    report.train_size, report.validation_size = len(train), len(validation)
    report.holdout_size = len(holdout)

    init = load_shipped_classifier() if warm_start else None
    trainer = AveragedPerceptronTrainer(init=init)
    best_accuracy, best_trainer = -1.0, trainer
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(train, rng)
        accuracy, _, _ = _holdout_accuracy(trainer.to_classifier(), validation)
        report.per_epoch_validation.append(round(accuracy, 4))
        if accuracy > best_accuracy:
            report.best_epoch = epoch
            best_accuracy, best_trainer = accuracy, _snapshot(trainer)
    new_classifier = best_trainer.to_classifier()

    new_accuracy, new_by_tier, new_confusion = _holdout_accuracy(new_classifier, holdout)
    shipped = load_shipped_classifier()
    shipped_accuracy, shipped_by_tier = None, {}
    if shipped is not None:
        shipped_accuracy, shipped_by_tier, _ = _holdout_accuracy(shipped, holdout)
    report.new_holdout_accuracy = new_accuracy
    report.new_holdout_by_tier = {k: round(v, 4) for k, v in new_by_tier.items()}
    report.new_confusion = new_confusion
    report.shipped_holdout_accuracy = shipped_accuracy
    report.shipped_holdout_by_tier = {k: round(v, 4) for k, v in shipped_by_tier.items()}

    # Platt re-fit on the same holdout, for the new model and (diagnostically)
    # for the shipped one.
    from llm_proxy.routing.decision.calibration import fit_platt_from_evals

    new_pairs = _platt_pairs(holdout, new_classifier)
    temperature_new = fit_platt_from_evals(new_pairs).temperature
    report.platt_temperature_new = temperature_new
    if shipped is not None:
        shipped_pairs = _platt_pairs(holdout, shipped)
        report.platt_temperature_shipped = fit_platt_from_evals(shipped_pairs).temperature

    report.no_baseline = shipped_accuracy is None
    report.gate_passed = shipped_accuracy is None or new_accuracy > shipped_accuracy
    if report.gate_passed and out_model is not None:
        save_classifier(new_classifier, out_model)
        report.model_written = str(out_model)
    if report.gate_passed and out_calibration is not None and temperature_new:
        out_calibration.parent.mkdir(parents=True, exist_ok=True)
        out_calibration.write_text(
            orjson.dumps({"temperature": temperature_new, "version": 1}).decode("utf-8"),
            encoding="utf-8",
        )
        report.calibration_written = str(out_calibration)
    return report


def _snapshot(trainer: AveragedPerceptronTrainer) -> AveragedPerceptronTrainer:
    """A deep copy of the trainer's current weight state."""
    copy = AveragedPerceptronTrainer(init=None)
    copy._weights = {
        tier: defaultdict(float, dict(trainer._weights[tier]))
        for tier in ScriptAgnosticClassifier.TIERS
    }
    copy._avg_weights = {
        tier: defaultdict(float, dict(trainer._avg_weights[tier]))
        for tier in ScriptAgnosticClassifier.TIERS
    }
    copy._update_count = trainer._update_count
    return copy


def render_report(report: TrainReport) -> str:
    """Human-readable training report."""
    lines = ["# Flywheel training report", ""]
    lines.append(f"- Labeled records: {report.examples} usable ({report.label_verdict_counts})")
    lines.append(f"- Sources: {report.label_source_counts}")
    lines.append(
        f"- Train/validation/holdout: {report.train_size}/"
        f"{report.validation_size}/{report.holdout_size}"
    )
    lines.append(
        f"- Validation accuracy per epoch (best epoch {report.best_epoch}): "
        f"{report.per_epoch_validation}"
    )
    lines.append(
        f"- NEW holdout accuracy: {_fmt(report.new_holdout_accuracy)} "
        f"(by tier: {report.new_holdout_by_tier})"
    )
    lines.append(
        f"- SHIPPED holdout accuracy: {_fmt(report.shipped_holdout_accuracy)} "
        f"(by tier: {report.shipped_holdout_by_tier})"
    )
    lines.append(f"- Confusion (new): {report.new_confusion}")
    lines.append(
        f"- Platt temperature new/shipped: {report.platt_temperature_new} / "
        f"{report.platt_temperature_shipped}"
    )
    if report.no_baseline:
        lines.append("- No shipped baseline found: the gate passes without a comparison")
    lines.append(
        "- Gate: "
        + ("PASSED — artifact written" if report.gate_passed else "NOT PASSED — nothing written")
    )
    if report.model_written:
        lines.append(f"- Model: {report.model_written}")
    if report.calibration_written:
        lines.append(f"- Calibration suggestion: {report.calibration_written}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


__all__ = [
    "AveragedPerceptronTrainer",
    "Example",
    "TrainReport",
    "build_training_features",
    "load_examples",
    "render_report",
    "save_classifier",
    "train_and_gate",
]
