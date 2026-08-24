"""Keyword-Free Classifier v7.

Architecture:
  Level 0: Structural trivial detection (token count, no keyword lists)
  Level 1: Model prediction on structural + Unicode + n-gram features
           N-grams learn keyword-equivalent patterns from training data
  Level 2: Structural-only fallback (when model is unavailable)

Feature groups (no keyword lists anywhere):
  - 12 structural scores (enumeration, sentences, code symbols, math, ...)
  - 15 Unicode block proportions (latin, cjk, hangul, arabic, ...)
  - ~8 context features (tools_present, conversation_depth, ...)
  - 4096 n-gram features (char 3-5 grams, learned from data)

All semantic signals come from n-grams trained on data, not hardcoded
keyword lists.  The model discovers which character patterns predict
difficulty — no manual vocabulary maintenance needed.
"""

import math
from pathlib import Path

import platformdirs

from llm_proxy.routing.assets import asset_path
from llm_proxy.routing.learned import ScriptAgnosticClassifier
from llm_proxy.routing.structural import (
    estimate_tokens,
    extract_structural_features,
    extract_unicode_block_features,
)
from llm_proxy.routing.types import (
    ScoringConfig,
    ScoringResult,
    Tier,
)

_model: ScriptAgnosticClassifier | None = None
_model_load_attempted = False


def _get_online_model_path() -> Path:
    data_dir = Path(platformdirs.user_data_dir("llm-proxy", "llm-proxy")) / "routing-assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "model_online.json"


def _merge_feature_text(prompt: str, system_prompt: str | None = None) -> str:
    system = str(system_prompt or "").strip()
    if not system:
        return prompt
    return f"{system}\n\n{prompt}"


def _ensure_model_loaded() -> None:
    global _model, _model_load_attempted
    if _model_load_attempted:
        return
    _model_load_attempted = True
    online = _get_online_model_path()
    default = asset_path("model.json")
    if online.exists():
        _model = ScriptAgnosticClassifier()
        _model.load(online)
    elif default.exists():
        _model = ScriptAgnosticClassifier()
        _model.load(default)


def _extract_all_features(
    prompt: str,
    system_prompt: str | None = None,
    context_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract the complete feature vector — no keywords.

    Structural features detect code/math/structure via symbols and
    character-level statistics.  N-grams learn vocabulary-equivalent
    patterns from training data.  Context features encode agentic
    step information as pure numerical signals.
    """
    feature_text = _merge_feature_text(prompt, system_prompt)
    struct_dims = extract_structural_features(feature_text)
    structural_scores = {d.name: d.score for d in struct_dims}

    unicode_blocks = extract_unicode_block_features(feature_text)

    if _model is not None:
        return _model._build_features(
            structural_scores,
            unicode_blocks,
            prompt=feature_text,
            context_features=context_features,
        )

    features: dict[str, float] = {}
    for name, score in structural_scores.items():
        features[f"s_{name}"] = score
    for name, prop in unicode_blocks.items():
        features[f"u_{name}"] = prop
    if context_features:
        for name, value in context_features.items():
            key = name if name.startswith("ctx_") else f"ctx_{name}"
            features[key] = value
    return features


# ─── Trivial Detection (structural only, no keyword lists) ───


def _check_trivial(prompt: str, tokens: int) -> Tier | None:
    """Detect trivially simple or trivially long prompts via structure only."""
    if tokens <= 1:
        return Tier.SIMPLE
    if tokens > 100_000:
        return Tier.COMPLEX
    stripped = prompt.strip()
    if (
        len(stripped) < 15
        and not any(c in stripped for c in "{}[]();=<>+-*/\\|@#$%^\u0026")
        and (stripped.endswith("?") or stripped.endswith("？"))
    ):
        return Tier.SIMPLE
    return None


def _soften_short_code_question(prompt: str, estimated_tokens: int, tier: Tier) -> Tier:
    """Short fenced-code prompts should not jump straight to COMPLEX."""
    if tier is not Tier.COMPLEX:
        return tier
    if estimated_tokens > 48:
        return tier
    if "```" not in prompt:
        return tier
    return Tier.MEDIUM


def _soften_low_structure_question(
    prompt: str,
    estimated_tokens: int,
    tier: Tier,
    features: dict[str, float],
) -> Tier:
    """Long natural-language questions should not be escalated by n-grams alone."""
    if tier is not Tier.COMPLEX:
        return tier
    if estimated_tokens > 80:
        return tier
    stripped = prompt.strip()
    if not (stripped.endswith("?") or stripped.endswith("？")):
        return tier
    if features.get("s_code_markers", 0.0) > 0.05:
        return tier
    if features.get("s_math_symbols", 0.0) > 0.05:
        return tier
    if features.get("s_nesting_depth", 0.0) > 0.05:
        return tier
    if features.get("s_enumeration_density", 0.0) > 0.10:
        return tier
    if features.get("s_requirement_phrases", 0.0) > 0.10:
        return tier
    if features.get("s_unique_concept_density", 0.0) > 0.10:
        return tier
    if features.get("s_avg_word_length", 0.0) >= 0.35:
        return tier
    if features.get("s_compression_complexity", 0.0) >= 0.70:
        return tier
    return Tier.MEDIUM


# ─── Structural-only fallback (when model unavailable) ───


def _sigmoid(distance: float, steepness: float) -> float:
    clamped = max(-50.0, min(50.0, steepness * distance))
    return 1.0 / (1.0 + math.exp(-clamped))


def _rule_based_classify(
    all_features: dict[str, float],
    config: ScoringConfig,
) -> tuple[Tier, float]:
    """Fallback classification using structural weights only."""
    sw = config.structural_weights
    weight_map = {
        "s_normalized_length": sw.normalized_length,
        "s_enumeration_density": sw.enumeration_density,
        "s_sentence_count": sw.sentence_count,
        "s_code_markers": sw.code_markers,
        "s_math_symbols": sw.math_symbols,
        "s_nesting_depth": sw.nesting_depth,
        "s_vocabulary_diversity": sw.vocabulary_diversity,
        "s_avg_word_length": sw.avg_word_length,
        "s_alphabetic_ratio": sw.alphabetic_ratio,
        "s_functional_intent": sw.functional_intent,
        "s_unique_concept_density": sw.unique_concept_density,
        "s_requirement_phrases": sw.requirement_phrases,
    }

    score = sum(all_features.get(k, 0.0) * w for k, w in weight_map.items())

    bounds = config.tier_boundaries
    if score < bounds.simple_medium:
        tier, dist = Tier.SIMPLE, bounds.simple_medium - score
    elif score < bounds.medium_complex:
        tier = Tier.MEDIUM
        dist = min(score - bounds.simple_medium, bounds.medium_complex - score)
    else:
        tier, dist = Tier.COMPLEX, score - bounds.medium_complex

    confidence = _sigmoid(dist, config.confidence_steepness)
    return tier, confidence


# ─── Main Entry ───


def classify(
    prompt: str,
    system_prompt: str | None = None,
    config: ScoringConfig | None = None,
    context_features: dict[str, float] | None = None,
) -> ScoringResult:
    if config is None:
        config = ScoringConfig()

    feature_text = _merge_feature_text(prompt, system_prompt)
    estimated_tokens = estimate_tokens(feature_text)
    _ensure_model_loaded()

    trivial = _check_trivial(feature_text, estimated_tokens)
    if trivial is not None:
        trivial_complexity = 0.0 if trivial is Tier.SIMPLE else 0.90
        return ScoringResult(
            tier=trivial,
            confidence=0.95,
            signals=(f"trivial:{trivial.value}",),
            complexity=trivial_complexity,
        )

    all_features = _extract_all_features(
        prompt,
        system_prompt=system_prompt,
        context_features=context_features,
    )

    if _model is not None:
        complexity, tier_str, confidence = _model.predict_complexity(all_features)
        normalized_tier = "COMPLEX" if tier_str == "REASONING" else tier_str
        tier = Tier(normalized_tier)
        tier = _soften_short_code_question(prompt, estimated_tokens, tier)
        tier = _soften_low_structure_question(prompt, estimated_tokens, tier, all_features)
        normalized_tier = tier.value
        if tier is Tier.MEDIUM and complexity > 0.40:
            complexity = 0.40
        complexity = max(0.0, min(1.0, complexity))
        signals = (f"model:{normalized_tier}({confidence:.2f})", f"complexity:{complexity:.2f}")
        return ScoringResult(
            tier=tier,
            confidence=confidence,
            signals=signals,
            complexity=complexity,
        )

    tier, confidence = _rule_based_classify(all_features, config)
    _TIER_TO_COMPLEXITY = {Tier.SIMPLE: 0.0, Tier.MEDIUM: 0.40, Tier.COMPLEX: 0.90}
    complexity = _TIER_TO_COMPLEXITY.get(tier, 0.33)
    struct_dims = extract_structural_features(prompt)
    signals = [d.signal for d in struct_dims if d.signal is not None]
    signals.append("rule-fallback")
    signals.append(f"complexity:{complexity:.2f}")

    if confidence < config.confidence_threshold:
        return ScoringResult(
            tier=None,
            confidence=confidence,
            signals=tuple(signals),
            dimensions=tuple(struct_dims),
            complexity=complexity,
        )

    return ScoringResult(
        tier=tier,
        confidence=confidence,
        signals=tuple(signals),
        dimensions=tuple(struct_dims),
        complexity=complexity,
    )
