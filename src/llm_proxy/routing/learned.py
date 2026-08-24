"""Script-Agnostic Learned Classifier v2.

Input features (all script-agnostic):
  1. 12 structural feature scores (enumeration, sentence_count, code, math, ...)
  2. 15 Unicode block proportions (latin, cjk, hangul, arabic, cyrillic, ...)
  3. 12 keyword feature scores (optional, for same-script languages)
  Total: ~39 named features → Averaged Perceptron

Why this works across scripts:
  - Structural features are universal (commas, sentences, brackets work everywhere)
  - Unicode block features tell the model WHAT SCRIPT without needing specific chars
  - The model learns: "high CJK + question mark + short = SIMPLE" for ALL CJK languages
  - Keyword features add bonus for languages with known vocabulary

Weights are loaded from a bundled asset; ``predict_complexity`` scores new
requests against the frozen weights.
"""

import math
from collections import defaultdict
from pathlib import Path

import orjson

FEATURE_DIM_NGRAM = 4096
NGRAM_RANGE = (3, 5)


def _signed_hash(s: str) -> tuple[int, float]:
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    sign = 1.0 if (h >> 16) & 1 == 0 else -1.0
    return h % FEATURE_DIM_NGRAM, sign


def _extract_ngram_features(text: str) -> dict[str, float]:
    """Char n-gram features with 'ngram_' prefix."""
    text_lower = text.lower().strip()
    features: dict[str, float] = defaultdict(float)
    for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        for i in range(len(text_lower) - n + 1):
            gram = text_lower[i : i + n]
            bucket, sign = _signed_hash(gram)
            features[f"ngram_{bucket}"] += sign
    norm = math.sqrt(sum(v * v for v in features.values())) or 1.0
    return {k: v / norm for k, v in features.items()}


class ScriptAgnosticClassifier:
    """Averaged Perceptron on structured features — generalizes across scripts.

    Feature groups:
      - structural_*: 12 scores from structural feature extractors
      - unicode_*: 15 Unicode block proportions
      - keyword_*: 12 scores from keyword extractors
      - ngram_*: char n-gram features (optional boost, low weight for unseen scripts)
    """

    TIERS = ("SIMPLE", "MEDIUM", "COMPLEX")

    def __init__(self, use_ngrams: bool = True) -> None:
        self._weights: dict[str, dict[str, float]] = {t: defaultdict(float) for t in self.TIERS}
        self._avg_weights: dict[str, dict[str, float]] = {t: defaultdict(float) for t in self.TIERS}
        self._update_count = 0
        self._trained = False
        self._use_ngrams = use_ngrams

    def _build_features(
        self,
        structural_scores: dict[str, float],
        unicode_blocks: dict[str, float],
        prompt: str = "",
        context_features: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Build the full feature vector from component parts."""
        features: dict[str, float] = {}

        for name, score in structural_scores.items():
            features[f"s_{name}"] = score

        for name, proportion in unicode_blocks.items():
            features[f"u_{name}"] = proportion

        if context_features:
            for name, value in context_features.items():
                key = name if name.startswith("ctx_") else f"ctx_{name}"
                features[key] = value

        if self._use_ngrams and prompt:
            ngram_feats = _extract_ngram_features(prompt)
            ngram_scale = 0.5
            for k, v in ngram_feats.items():
                features[k] = v * ngram_scale

        return features

    def _score_raw(self, features: dict[str, float], use_avg: bool = True) -> dict[str, float]:
        weights = self._avg_weights if use_avg and self._update_count > 0 else self._weights
        scores: dict[str, float] = {}
        for tier in self.TIERS:
            w = weights[tier]
            scores[tier] = sum(val * w.get(feat, 0.0) for feat, val in features.items())
        return scores

    COMPLEXITY_ANCHORS = {"SIMPLE": 0.0, "MEDIUM": 0.40, "COMPLEX": 0.90}

    def predict_complexity(self, features: dict[str, float]) -> tuple[float, str, float]:
        """Return ``(complexity, tier, confidence)``.

        ``complexity`` is a continuous 0.0–1.0 score derived from the
        softmax probability-weighted tier anchors.  It replaces discrete
        tier buckets with a smooth value that the selector can use to
        interpolate scoring weights.
        """
        if not self._trained:
            return (0.33, "MEDIUM", 0.0)

        scores = self._score_raw(features, use_avg=True)
        max_s = max(scores.values())
        exp_scores = {t: math.exp(min(s - max_s, 50)) for t, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {t: e / total for t, e in exp_scores.items()}

        complexity = sum(probs[t] * self.COMPLEXITY_ANCHORS[t] for t in self.TIERS)
        complexity = max(0.0, min(1.0, complexity))

        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return (complexity, best, probs[best])

    @classmethod
    def _collapse_loaded_weights(
        cls,
        raw_weights: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {tier: dict(raw_weights.get(tier, {})) for tier in cls.TIERS}

    def load(self, path: Path) -> None:
        data = orjson.loads(path.read_text())
        avg_weights = self._collapse_loaded_weights(data.get("avg_weights", {}))
        weights = self._collapse_loaded_weights(data.get("weights", {}))
        self._avg_weights = {t: defaultdict(float, avg_weights.get(t, {})) for t in self.TIERS}
        self._weights = {t: defaultdict(float, weights.get(t, {})) for t in self.TIERS}
        self._update_count = data.get("update_count", 1)
        self._use_ngrams = data.get("use_ngrams", True)
        self._trained = True
