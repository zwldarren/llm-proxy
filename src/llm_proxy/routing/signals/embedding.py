"""Signal C: embedding-based tier prediction.

Uses frozen bge-small embeddings with either:
  1. A trained classifier (XGBoost) — loaded from the vendored
     embedding_classifier.ubj asset (with a legacy .pkl fallback)
  2. KNN fallback — distance-weighted vote of K nearest neighbors
"""

import asyncio
import contextlib
import logging
import pickle
import sys
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from llm_proxy.routing.assets import asset_path
from llm_proxy.routing.signal_tuning import strip_client_wrapper_blocks
from llm_proxy.routing.signals.base import TierVote

logger = logging.getLogger(__name__)

K_NEIGHBORS = 7
MIN_CONFIDENCE_TO_VOTE = 0.3

_cache: dict[str, EmbeddingSignal | None] = {}
_load_lock_obj: asyncio.Lock | None = None
_load_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_load_lock() -> asyncio.Lock:
    global _load_lock_obj, _load_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _load_lock_obj is None or _load_lock_loop != loop:
        # Clear the cached singleton when the event loop changes so that
        # the next call to get_embedding_signal re-initialises in the new
        # loop context (e.g. after a reload or in tests).
        _cache.clear()
        _load_lock_obj = asyncio.Lock()
        _load_lock_loop = loop
    return _load_lock_obj


def _import_onnx_deps():
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The smart-routing embedding signal requires the 'smart-routing' extra. "
            "Install it with: uv sync --extra smart-routing"
        ) from exc
    return AutoTokenizer, ort, hf_hub_download


def _default_embed_fn():
    AutoTokenizer, ort, hf_hub_download = _import_onnx_deps()

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Download ONNX model files (graph + external weights)
    model_path = hf_hub_download(
        repo_id="onnx-community/bge-small-en-v1.5-ONNX",
        filename="onnx/model.onnx",
    )
    # Ensure the external data file is also cached in the same directory
    hf_hub_download(
        repo_id="onnx-community/bge-small-en-v1.5-ONNX",
        filename="onnx/model.onnx_data",
    )

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    # BERT model: input names are typically input_ids, attention_mask, token_type_ids
    input_names = {inp.name for inp in session.get_inputs()}

    def _embed(text: str) -> np.ndarray:
        tokens = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed: dict[str, np.ndarray] = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        if "token_type_ids" in input_names:
            tti = tokens.get(
                "token_type_ids",
                np.zeros_like(tokens["input_ids"]),
            )
            feed["token_type_ids"] = tti

        outputs = session.run(None, feed)
        # CLS pooling: take the first token's embedding
        embedding = outputs[0][0, 0, :]
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return np.asarray(embedding, dtype=np.float32)

    return _embed


async def get_embedding_signal(app_state) -> EmbeddingSignal | None:
    """Lazy singleton; returns None if ML deps unavailable (graceful degradation).

    Model loading runs in a thread via ``asyncio.to_thread`` so the event loop
    is not blocked during download or model initialisation.
    """
    if "embedding_signal" in _cache:
        return _cache["embedding_signal"]

    async with _get_load_lock():
        # Double-check after acquiring the lock to avoid redundant loading.
        if "embedding_signal" in _cache:
            return _cache["embedding_signal"]
        try:
            embed_fn = await asyncio.to_thread(_default_embed_fn)
            sig = EmbeddingSignal(embed_fn=embed_fn)
        except Exception:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "Embedding signal unavailable; routing degrades to signals A+B.", exc_info=True
            )
            sig = None
        _cache["embedding_signal"] = sig
        return sig


def _classifier_load_warning(path: Path, error: Exception) -> str:
    detail = str(error).strip() or error.__class__.__name__
    hint = ""
    marker = f"{error.__class__.__name__} {detail}".lower()
    if "libomp" in marker:
        hint = (
            " On macOS, install the OpenMP runtime with `brew install libomp`, "
            "then restart llm-proxy."
        )
    elif isinstance(error, ImportError):
        hint = (
            " Install the 'smart-routing' extra to enable the embedding classifier: "
            "uv sync --extra smart-routing"
        )
    return (
        f"Embedding classifier failed to load from {path}: {detail}. "
        "Routing will continue with weaker metadata/structural signals until this is fixed."
        f"{hint}"
    )


def _warn_classifier_load_failure(path: Path, error: Exception) -> None:
    message = _classifier_load_warning(path, error)
    logger.warning(message)
    print(f"[llm-proxy] Warning: {message}", file=sys.stderr)


def _normalize_content(content: Any) -> str:
    """Normalize message content — handles string, list, and dataclass formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        has_multimodal = False
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    parts.append(part.get("text", ""))
                elif part_type in {
                    "input_image",
                    "image",
                    "image_url",
                    "input_file",
                    "input_audio",
                    "input_video",
                }:
                    has_multimodal = True
                elif part_type == "input_text":
                    parts.append(part.get("text", ""))
                elif part_type == "tool_result":
                    # Anthropic tool_result: {'type': 'tool_result', 'content': ...}
                    # OpenAI function_call_output: {'type': 'function_call_output', 'output': ...}
                    result_content = part.get("content", "")
                    if isinstance(result_content, str):
                        parts.append(result_content)
                    elif isinstance(result_content, list):
                        for item in result_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                elif part_type == "input_text":
                    parts.append(part.get("text", ""))
            elif not isinstance(part, (dict, str)) and hasattr(part, "text"):
                parts.append(getattr(part, "text", ""))
            elif isinstance(part, str):
                parts.append(part)
        result = " ".join(parts)
        # If we have multimodal content but no text, add a marker so signals
        # don't treat this as empty text (which would cause them to abstain)
        if not result and has_multimodal:
            return "[multimodal_content]"
        return result
    # Single dataclass object with a text attribute (e.g. TextBlock)
    if not isinstance(content, (dict, str)) and hasattr(content, "text"):
        return str(getattr(content, "text", ""))
    # Handle dict with type field
    if isinstance(content, dict):
        content_type = content.get("type")
        if content_type == "text":
            return content.get("text", "")
        if content_type in {"input_text", "input_image", "image", "image_url"}:
            return content.get("text", "") or content.get("image_url", "") or ""
        if content_type == "tool_result":
            # Anthropic tool_result format
            result_content = content.get("content", "")
            if isinstance(result_content, str):
                return result_content
    return str(content) if content else ""


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            text = _normalize_content(m.get("content", ""))
            stripped = strip_client_wrapper_blocks(text)
            return stripped if stripped else text
    return ""


def _tool_call_preview(tool_call: dict[str, Any], budget_chars: int = 320) -> str:
    fn = tool_call.get("function") or {}
    if not isinstance(fn, dict):
        return ""
    name = str(fn.get("name") or tool_call.get("name") or "?").strip() or "?"
    arguments = fn.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = orjson.loads(arguments)
        except orjson.JSONDecodeError:
            arg_text = arguments
        else:
            if isinstance(parsed, dict):
                command = parsed.get("command")
                if isinstance(command, str):
                    arg_text = command
                else:
                    arg_text = orjson.dumps(parsed, option=orjson.OPT_SORT_KEYS).decode()
            else:
                arg_text = str(parsed)
    elif arguments is None:
        arg_text = ""
    else:
        arg_text = str(arguments)
    arg_text = " ".join(arg_text.split())
    if len(arg_text) > budget_chars:
        arg_text = arg_text[:budget_chars] + "..."
    return f"{name}({arg_text})" if arg_text else name


def _extract_agent_state(messages: list[dict[str, Any]], budget_chars: int = 2400) -> str:
    """Second embedding input: recent agent state.

    For multi-step agent trajectories the last_user text is often identical across
    steps (e.g. a swebench PR description), causing embedding collisions. The
    tier-discriminating signal lives in the recent assistant/tool exchange —
    what's been tried, what errors appeared, which tools are being used. We
    embed that separately and concatenate both vectors at classifier time.
    """
    recent: list[str] = []
    for m in reversed(messages):
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        content = _normalize_content(m.get("content", ""))
        tcs = m.get("tool_calls") or []
        if tcs:
            previews = [
                preview
                for tc in tcs[:4]
                if isinstance(tc, dict)
                for preview in [_tool_call_preview(tc)]
                if preview
            ]
            if previews:
                tool_text = "; ".join(previews)
                content = (
                    f"{content}\n[tool_calls: {tool_text}]"
                    if content
                    else f"[tool_calls: {tool_text}]"
                )
        if content:
            # Keep enough recent state to preserve intent across a short read /
            # verify sequence, while still bounding the classifier input.
            recent.append(f"[{role}] {content[:650]}")
        if len(recent) >= 8:
            break
    if not recent:
        return ""
    return " | ".join(reversed(recent))[:budget_chars]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


class EmbeddingSignal:
    def __init__(
        self,
        index_path: Path | None = None,
        labels_path: Path | None = None,
        model_name: str | None = "BAAI/bge-small-en-v1.5",
        classifier_path: Path | None = None,
        use_classifier: bool = True,
        classifier_fallback_threshold: float = 0.92,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ):
        self._embeddings: Any = None
        self._labels: list[int] | None = None
        self._embed_fn: Callable[[str], np.ndarray] | None = embed_fn
        self._classifier: Any = None  # sklearn classifier (optional)
        self._meta_scaler: Any = None  # StandardScaler for metadata features
        self._clf_fallback_threshold = classifier_fallback_threshold

        if index_path is None:
            index_path = asset_path("seed_embeddings.npy")
        if labels_path is None:
            labels_path = asset_path("seed_labels.json")
        if classifier_path is None:
            classifier_path = asset_path("embedding_classifier.ubj")

        if index_path.exists() and labels_path.exists():
            self._embeddings = np.load(index_path)
            with open(labels_path, encoding="utf-8") as f:
                self._labels = orjson.loads(f.read())
            logger.info("Loaded embedding index: %s vectors", len(self._labels))

        # Try to load trained classifier (skip entirely when use_classifier=False)
        if use_classifier:
            self._load_classifier(classifier_path)

        if self._embed_fn is None and model_name:
            try:
                self._embed_fn = _default_embed_fn()
            except ImportError as e:
                logger.warning(
                    "ONNX embedding deps not installed; embedding signal will abstain (%s)", e
                )
            except Exception as e:
                logger.warning("Failed to load embedding model %s: %s", model_name, e)

    def _load_classifier(self, classifier_path: Path) -> None:
        """Load the trained embedding classifier (XGBoost).

        Prefers the native XGBoost format (``embedding_classifier.ubj``), which
        loads without warnings. Falls back to a legacy pickled estimator
        (``embedding_classifier.pkl``) for deployments created before the
        migration, converting it to the native format on the fly.

        A ``.ubj`` file that fails native loading is retried as a pickle: some
        deployments copied the asset before the native-format migration, when
        the file was a pickled estimator that only carried the ``.ubj`` name.
        """
        native = (
            classifier_path
            if classifier_path.suffix == ".ubj"
            else classifier_path.with_suffix(".ubj")
        )
        legacy = native.with_suffix(".pkl")

        if native.exists():
            try:
                import xgboost as xgb

                self._classifier = xgb.XGBClassifier()
                self._classifier.load_model(native)
                logger.info("Loaded trained embedding classifier from %s", native)
                self._try_load_scaler(native.parent)
                return
            except Exception as e:
                # The file may be a legacy pickled estimator that was renamed
                # to .ubj without conversion. Load it via pickle and convert
                # to the native format so the next startup loads natively.
                try:
                    with warnings.catch_warnings():
                        # XGBoost warns when unpickling models saved by an
                        # older version. The file is known-good and is
                        # converted to the native format below, so the
                        # warning is noise here.
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*loading a serialized model.*",
                            category=UserWarning,
                        )
                        with open(native, "rb") as f:
                            self._classifier = pickle.load(f)
                    logger.info(
                        "Loaded pickled embedding classifier from %s; converting to native format",
                        native,
                    )
                    self._try_load_scaler(native.parent)
                    with contextlib.suppress(Exception):  # best-effort migration
                        self._classifier.save_model(native)
                    return
                except Exception:
                    # Both native and pickle loading failed. Reset the classifier
                    # so predict() degrades to KNN instead of crashing on the
                    # empty XGBClassifier left behind by the failed load_model.
                    self._classifier = None
                    _warn_classifier_load_failure(native, e)
                    return

        if legacy.exists():
            try:
                with warnings.catch_warnings():
                    # XGBoost warns when unpickling models saved by an older
                    # version. The legacy asset is known-good and is converted
                    # to the native format below, so the warning is noise here.
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*loading a serialized model.*",
                        category=UserWarning,
                    )
                    with open(legacy, "rb") as f:
                        self._classifier = pickle.load(f)
                logger.info("Loaded legacy pickled embedding classifier from %s", legacy)
                self._try_load_scaler(legacy.parent)
                with contextlib.suppress(Exception):  # best-effort migration
                    self._classifier.save_model(native)
            except Exception as e:
                self._classifier = None
                _warn_classifier_load_failure(legacy, e)

    def _try_load_scaler(self, directory: Path) -> None:
        """Load metadata feature scaler if available."""
        scaler_path = directory / "meta_scaler.pkl"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    self._meta_scaler = pickle.load(f)
                logger.info("Loaded metadata scaler from %s", scaler_path)
            except Exception:
                logger.warning(
                    "Failed to load metadata scaler from %s; "
                    "embedding routing signal will run in degraded mode",
                    scaler_path,
                    exc_info=True,
                )

    @staticmethod
    def _extract_meta_features(messages: list[dict[str, Any]], text: str) -> list[float]:
        """Extract metadata features (15 dims) to augment embedding vector.

        The classifier assigns ~22% of its decision weight to these features —
        the raw BGE embedding alone saturates on multi-step agent conversations
        that share identical last-user text, so these structural/conversational
        cues are what separates step_2 from step_7 of the same issue.
        """
        msg_count = len(messages)
        has_tools = int(any(m.get("role") == "tool" or m.get("tool_calls") for m in messages))
        tool_count = sum(1 for m in messages if m.get("role") == "tool" or m.get("tool_calls"))
        user_len = len(text)
        user_words = len(text.split())
        has_code = int("```" in text)
        has_question = int("?" in text[-50:] if len(text) > 50 else "?" in text)

        # Conversation-state features for step-within-issue discrimination
        user_msg_count = sum(1 for m in messages if m.get("role") == "user")
        step_proxy = max(0, user_msg_count - 1)

        tool_error_count = 0
        for m in messages:
            if m.get("role") != "tool":
                continue
            c = _normalize_content(m.get("content", "")).lower()
            if any(k in c for k in ("error", "traceback", "exception", "failed")):
                tool_error_count += 1

        tool_names: list[str] = []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name") or tc.get("name", "")
                if name:
                    tool_names.append(name)
        unique_tool_names = len(set(tool_names))
        tn_counts = Counter(tool_names)
        max_tool_repeat = max(tn_counts.values()) if tn_counts else 0
        has_retry = int(max_tool_repeat >= 3)

        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assistant_msg_count = len(assistant_msgs)
        if assistant_msgs:
            lens = [len(_normalize_content(m.get("content", ""))) for m in assistant_msgs]
            avg_assistant_len = sum(lens) / len(lens)
        else:
            avg_assistant_len = 0.0
        has_code_in_assistant = int(
            any("```" in _normalize_content(m.get("content", "")) for m in assistant_msgs)
        )

        return [
            msg_count,
            has_tools,
            tool_count,
            user_len,
            user_words,
            has_code,
            has_question,
            step_proxy,
            tool_error_count,
            unique_tool_names,
            max_tool_repeat,
            has_retry,
            avg_assistant_len,
            assistant_msg_count,
            has_code_in_assistant,
        ]

    def predict(self, row: dict[str, Any]) -> TierVote:
        if self._embed_fn is None:
            return TierVote(tier_id=None, confidence=0.0)

        messages = row.get("messages", [])
        text = _extract_last_user_message(messages)
        if not text.strip():
            return TierVote(tier_id=None, confidence=0.0)

        query_vec = self._embed_fn(text)
        meta_feats = self._extract_meta_features(messages, text)

        # If the classifier was trained on dual embeddings (v0 = last_user,
        # v1 = agent_state), compute the second embedding too. We detect this
        # by the classifier's n_features_in_: 399 = single (384 + 15), 783 = dual.
        state_vec = None
        if self._classifier is not None:
            try:
                expected = int(self._classifier.n_features_in_)
            except Exception:  # noqa: BLE001 - empty/unfitted booster; treat as no classifier
                expected = 0
            single_dim = int(query_vec.shape[-1]) + len(meta_feats)
            if expected == single_dim + int(query_vec.shape[-1]):
                state_text = _extract_agent_state(messages)
                # Fall back to duplicating the user embedding so dim matches.
                state_vec = self._embed_fn(state_text) if state_text else query_vec

        # Hybrid: classifier first, KNN fallback when classifier is uncertain.
        if self._classifier is not None:
            vote = self._predict_classifier(query_vec, meta_feats, state_vec=state_vec)
            if vote.confidence >= self._clf_fallback_threshold:
                return vote
            if self._embeddings is not None and self._labels is not None:
                knn_vote = self._predict_knn(query_vec)
                # When kNN abstains (e.g., embedding collision with mixed labels),
                # trust the classifier's meta-feature-aware prediction rather
                # than returning None. This matters for multi-step swebench rows
                # where the last_user text is identical across steps but the
                # per-step difficulty varies.
                if knn_vote.tier_id is None and vote.tier_id is not None:
                    return vote
                return knn_vote
            return vote

        # No classifier — KNN only
        if self._embeddings is None or self._labels is None:
            return TierVote(tier_id=None, confidence=0.0)
        return self._predict_knn(query_vec)

    def _predict_classifier(
        self,
        query_vec: np.ndarray,
        meta_feats: list[float] | None = None,
        state_vec: np.ndarray | None = None,
    ) -> TierVote:
        """Predict using classifier on embedding + metadata features.

        If meta_scaler is missing but classifier was trained on combined features,
        skip classifier entirely and return low-confidence abstain to trigger KNN fallback.
        """
        vec = query_vec.reshape(1, -1)
        if state_vec is not None:
            vec = np.hstack([vec, state_vec.reshape(1, -1)])
        if meta_feats is not None and self._meta_scaler is not None:
            meta_arr = np.array([meta_feats], dtype=float)
            meta_scaled = self._meta_scaler.transform(meta_arr)
            vec = np.hstack([vec, meta_scaled])
        elif self._meta_scaler is None and hasattr(self._classifier, "n_features_in_"):
            if self._classifier.n_features_in_ > vec.shape[1]:
                # Classifier expects combined features but scaler is missing —
                # can't match dimensions
                return TierVote(tier_id=None, confidence=0.0)
        pred = int(self._classifier.predict(vec)[0])
        proba = self._classifier.predict_proba(vec)[0]
        confidence = float(proba[pred])

        if confidence < MIN_CONFIDENCE_TO_VOTE:
            return TierVote(tier_id=None, confidence=confidence)
        return TierVote(tier_id=pred, confidence=confidence)

    def _predict_knn(self, query_vec: np.ndarray) -> TierVote:
        """Predict using KNN distance-weighted vote."""
        assert self._embeddings is not None and self._labels is not None
        sims = _cosine_similarity(query_vec, self._embeddings)

        k = min(K_NEIGHBORS, len(self._labels))
        top_k_idx = np.argsort(sims)[-k:][::-1]
        top_k_sims = sims[top_k_idx]
        top_k_labels = [self._labels[i] for i in top_k_idx]

        # If top neighbors aren't similar enough, abstain — the query is
        # out-of-distribution relative to our training set.
        avg_top_sim = float(np.mean(top_k_sims[:3]))
        if avg_top_sim < 0.60:
            return TierVote(tier_id=None, confidence=0.0)

        # Embedding-collision abstain: when top-K have near-identical similarity
        # but mixed tier labels, the query text matches multiple rows whose
        # *steps* differ (same PR description, different mid-task state). The
        # kNN vote here is noise — abstain so the classifier (which sees meta
        # features) wins the ensemble.
        if float(top_k_sims[0]) > 0.995 and len(set(top_k_labels[:3])) > 1:
            return TierVote(tier_id=None, confidence=0.0)

        tier_scores: dict[int, float] = {}
        for label, sim in zip(top_k_labels, top_k_sims, strict=True):
            tier_scores[label] = tier_scores.get(label, 0.0) + max(0.0, float(sim))

        if not tier_scores:
            return TierVote(tier_id=None, confidence=0.0)

        total = sum(tier_scores.values())
        best_tier = max(tier_scores, key=lambda t: tier_scores[t])
        confidence = tier_scores[best_tier] / total if total > 0 else 0.0

        if confidence < MIN_CONFIDENCE_TO_VOTE:
            return TierVote(tier_id=None, confidence=confidence)

        return TierVote(tier_id=best_tier, confidence=confidence)
