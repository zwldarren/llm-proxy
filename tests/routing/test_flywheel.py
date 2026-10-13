"""Tests for the label flywheel: pair resolution, sources, and the trainer."""

import json
from pathlib import Path

import pytest

from llm_proxy.routing.flywheel.label import dump_items, load_items
from llm_proxy.routing.flywheel.pairs import (
    PAIR_BALANCED_VS_WEAK,
    PAIR_STRONG_VS_BALANCED,
    VoteOutcome,
    build_pairwise_questions,
    build_pairwise_state,
    parse_pairwise_answers,
    resolve_pair,
    resolve_verdict,
    truncate_head_tail,
)
from llm_proxy.routing.flywheel.sources import (
    PromptItem,
    iter_corpus_prompts,
    iter_request_log_prompts,
)
from llm_proxy.routing.flywheel.store import LabelRecord, LabelStore
from llm_proxy.routing.flywheel.train import (
    AveragedPerceptronTrainer,
    Example,
    build_training_features,
    train_and_gate,
)
from llm_proxy.routing.types import Tier

# ── pairwise rubric ──


def test_pairwise_questions_shape():
    questions = build_pairwise_questions()
    assert set(questions) == {"better"}
    assert questions["better"]["type"] == "choice"
    assert set(questions["better"]["criteria"]) == {"a", "b", "tie", "ambiguous"}


def test_build_pairwise_state_bounces_answers():
    state = build_pairwise_state(
        [{"role": "user", "content": "What is the capital of Portugal?"}],
        "Lisbon",
        "The capital city of Portugal is Lisbon.",
    )
    assert state["answer_a"] == "Lisbon"
    assert "capital of Portugal" in state["current_request"]
    assert state["request_flags"]["estimated_context_tokens"] > 0


def test_parse_pairwise_answers_tolerates_garbage():
    choice, probabilities = parse_pairwise_answers(
        {"better": {"choice": "a", "probabilities": {"a": 2, "b": 1, "tie": 1}}}
    )
    assert choice == "a"
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert parse_pairwise_answers(None)[0] is None
    assert parse_pairwise_answers({"better": {"choice": "nonsense"}})[0] is None
    assert parse_pairwise_answers({"other": {}})[0] is None


def test_truncate_head_tail_keeps_both_ends():
    text = "HEAD" + "x" * 5000 + "TAIL"
    cut = truncate_head_tail(text, 200)
    assert cut.startswith("HEAD")
    assert cut.endswith("TAIL")
    assert "middle omitted" in cut
    assert len(cut) <= 220


# ── pair and verdict resolution ──


def _vote(choice: str | None, error: str | None = None) -> VoteOutcome:
    return VoteOutcome(choice=choice)


def test_resolve_pair_requires_swap_agreement():
    # First call: high role presented as A. Second: swapped.
    assert resolve_pair(_vote("a"), _vote("b")) == "a"
    assert resolve_pair(_vote("b"), _vote("a")) == "b"
    assert resolve_pair(_vote("tie"), _vote("tie")) == "tie"
    # Position bias (a wins both times) is a contradiction -> abstain.
    assert resolve_pair(_vote("a"), _vote("a")) == "abstain"
    assert resolve_pair(_vote(None), _vote("b")) == "abstain"
    assert resolve_pair(_vote("ambiguous"), _vote("b")) == "abstain"


def test_resolve_verdict_chain():
    complex_chain = resolve_verdict({PAIR_STRONG_VS_BALANCED: "a", PAIR_BALANCED_VS_WEAK: "a"})
    assert complex_chain == ("COMPLEX", "strong>balanced>weak")
    assert (
        resolve_verdict({PAIR_STRONG_VS_BALANCED: "tie", PAIR_BALANCED_VS_WEAK: "a"})[0] == "MEDIUM"
    )
    assert (
        resolve_verdict({PAIR_STRONG_VS_BALANCED: "tie", PAIR_BALANCED_VS_WEAK: "tie"})[0]
        == "SIMPLE"
    )
    # Inconsistent chain: strong > balanced but balanced = weak.
    verdict, chain = resolve_verdict({PAIR_STRONG_VS_BALANCED: "a", PAIR_BALANCED_VS_WEAK: "tie"})
    assert verdict is None
    assert chain is not None
    assert resolve_verdict({PAIR_STRONG_VS_BALANCED: "abstain", PAIR_BALANCED_VS_WEAK: "a"}) == (
        None,
        None,
    )


# ── sources ──


def test_iter_corpus_prompts(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "c1",
                "messages": [{"role": "user", "content": "hi there friend"}],
                "expected_tier": "SIMPLE",
            }
        )
        + "\nnot json\n"
        + json.dumps({"id": "c2", "messages": [], "expected_tier": "MEDIUM"})
        + "\n",
        encoding="utf-8",
    )
    items = list(iter_corpus_prompts(path))
    assert [i.source_id for i in items] == ["c1"]
    assert items[0].authored_tier == "SIMPLE"
    assert items[0].source == "corpus:corpus"


def test_iter_request_log_prompts_filters_and_dedupes(tmp_path: Path):
    import sqlite3

    db = tmp_path / "config.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE request_logs (request_id TEXT, endpoint TEXT, status_code INT, "
        "timestamp REAL, request_body TEXT)"
    )
    good = {
        "messages": [
            {"role": "user", "content": "Explain how a transformer attention head works, please."}
        ]
    }
    rows = [
        ("r1", "/v1/chat/completions", 200, 100.0, json.dumps(good)),
        ("r2", "/v1/chat/completions", 200, 101.0, json.dumps(good)),  # duplicate content
        ("r3", "/v1/chat/completions", 500, 102.0, json.dumps(good)),  # failed status
        ("r4", "/v1/systemone", 200, 103.0, json.dumps(good)),  # judge shadow row
        ("r5", "/v1/chat/completions", 200, 104.0, "BODIES_DISABLED_MARKER"),
        (
            "r6",
            "/v1/chat/completions",
            200,
            105.0,
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        ),  # too short
    ]
    connection.executemany("INSERT INTO request_logs VALUES (?,?,?,?,?)", rows)
    connection.commit()
    connection.close()

    items = list(iter_request_log_prompts(db, limit=10))
    # Newest-first sampling: among the two identical-content rows, r2 (newer)
    # is kept and r1 is dropped as a duplicate.
    assert [i.source_id for i in items] == ["r2"]
    assert items[0].source == "request_logs"
    assert items[0].authored_tier is None


def test_prompt_item_store_round_trip(tmp_path: Path):
    items = [
        PromptItem(
            source="corpus:x",
            source_id="a",
            messages=[{"role": "user", "content": "hello"}],
            authored_tier="SIMPLE",
        ),
        PromptItem(
            source="request_logs", source_id="r9", messages=[{"role": "user", "content": "hi"}]
        ),
    ]
    path = tmp_path / "prompts.jsonl"
    dump_items(items, path)
    loaded = load_items(path)
    assert loaded == items


# ── store ──


def test_label_store_round_trip_and_resume(tmp_path: Path):
    store = LabelStore(tmp_path / "labels.jsonl")
    record = LabelRecord(
        source="corpus:x",
        source_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        models={"strong": "glm-5.3"},
        answers={"strong": "answer"},
        verdict_tier="SIMPLE",
        verdict_chain="strong=balanced=weak",
        ensemble_tier="MEDIUM",
        ensemble_confidence_raw=0.42,
        authored_tier="COMPLEX",
    )
    store.append(record)
    store.append(LabelRecord(source="corpus:x", source_id="torn", messages=[]))
    with store.path.open("ab") as handle:  # a torn line must not sink the file
        handle.write(b'{"source": "corpus:x", "source_id": "torn2", "messages": [')
    loaded = store.read_all()
    assert len(loaded) == 2
    assert loaded[0].verdict_tier == "SIMPLE"
    assert loaded[0].ensemble_confidence_raw == 0.42
    assert store.seen_source_ids() == {"corpus:x:c1", "corpus:x:torn"}


# ── trainer ──


def _example(tier: str, marker: float, index: int) -> Example:
    # One-hot per class: strictly separable with wide margins, so the averaged
    # perceptron converges instead of oscillating on a slim margin.
    feature = {"SIMPLE": "s_alpha", "MEDIUM": "s_beta", "COMPLEX": "s_gamma"}[tier]
    return Example(
        features={feature: 1.0, "s_alphabetic_ratio": 1.0},
        tier=tier,
        source="test",
        source_id=f"e{index}",
    )


def test_averaged_perceptron_learns_separable_toy():
    examples = (
        [_example("SIMPLE", 0.1, i) for i in range(30)]
        + [_example("MEDIUM", 0.5, i) for i in range(30)]
        + [_example("COMPLEX", 1.2, i) for i in range(30)]
    )
    trainer = AveragedPerceptronTrainer(init=None)
    import random

    rng = random.Random(7)
    for _ in range(30):
        trainer.train_epoch(examples, rng)
    classifier = trainer.to_classifier()
    for tier, feature in (("SIMPLE", "s_alpha"), ("MEDIUM", "s_beta"), ("COMPLEX", "s_gamma")):
        _, predicted, confidence = classifier.predict_complexity(
            {feature: 1.0, "s_alphabetic_ratio": 1.0}
        )
        assert predicted == tier
        assert confidence > 0.55


def test_training_features_match_request_path_shape():
    record = LabelRecord(
        source="corpus:x",
        source_id="c1",
        messages=[
            {"role": "system", "content": "Reply in JSON."},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ],
        verdict_tier="SIMPLE",
    )
    features = build_training_features(record)
    assert features is not None
    assert any(name.startswith("s_") for name in features)
    assert any(name.startswith("u_") for name in features)
    assert any(name.startswith("ngram_") for name in features)
    empty = LabelRecord(
        source="corpus:x",
        source_id="c2",
        messages=[{"role": "assistant", "content": "only an assistant turn"}],
        verdict_tier="SIMPLE",
    )
    assert build_training_features(empty) is None


def test_train_and_gate_writes_only_when_it_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Stub the ensemble read so the gate runs hermetically: the ensemble
    # mirrors the trained classifier's tier with a fixed raw confidence.
    from llm_proxy.routing.types import Tier

    def fake_classify(messages, features=None, **_):
        text = ""
        for message in messages or []:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                text = content if isinstance(content, str) else ""
        index = 0 if "marker0" in text else 1 if "marker1" in text else 2
        return _FakeResult(list(Tier)[index])

    monkeypatch.setattr("llm_proxy.routing.api.classify", fake_classify)

    records = [
        LabelRecord(
            source="test",
            source_id=f"c{i}",
            # Tier-distinct vocabularies make the n-gram features strongly
            # separable; the marker token is what the ensemble stub keys on.
            messages=[
                {
                    "role": "user",
                    "content": f"marker{i % 3}token "
                    + ("apple " if i % 3 == 0 else "banana " if i % 3 == 1 else "canary ") * 40,
                }
            ],
            verdict_tier=("SIMPLE", "MEDIUM", "COMPLEX")[i % 3],
        )
        for i in range(120)
    ]
    out_model = tmp_path / "model_online.json"
    out_cal = tmp_path / "calibration.json"
    report = train_and_gate(
        records,
        holdout_fraction=0.25,
        seed=1,
        epochs=8,
        warm_start=False,
        out_model=out_model,
        out_calibration=out_cal,
    )
    # The toy problem is separable by construction, so the gate passes and both
    # artifacts land.
    assert report.examples == 120
    assert report.gate_passed
    assert out_model.exists()
    assert out_cal.exists()
    assert json.loads(out_cal.read_text())["version"] == 1
    saved = json.loads(out_model.read_text())
    assert set(saved) >= {"avg_weights", "weights", "update_count", "use_ngrams"}
    assert set(saved["avg_weights"]) == {"SIMPLE", "MEDIUM", "COMPLEX"}


class _FakeResult:
    def __init__(self, tier: Tier) -> None:
        from types import SimpleNamespace

        self.v2 = SimpleNamespace(tier_id=list(Tier).index(tier), confidence=0.8, signals_text=())


# ── effort ladder ──


def test_parse_efforts():
    from llm_proxy.routing.flywheel.__main__ import _parse_efforts

    assert _parse_efforts(None) == {}
    assert _parse_efforts("weak=low,balanced=high,strong=max") == {
        "weak": "low",
        "balanced": "high",
        "strong": "max",
    }
    with pytest.raises(SystemExit):
        _parse_efforts("weak=low,chief=max")


def test_generation_body_carries_reasoning_effort():
    import asyncio

    from llm_proxy.routing.flywheel.generate import GenerationClient

    captured: dict = {}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Client:
        async def post(self, url, json=None, headers=None):
            captured["json"] = json
            return _Response()

    client = GenerationClient(base_url="http://x", api_key="k")
    client._client = _Client()
    outcome = asyncio.run(
        client.generate(
            [{"role": "user", "content": "hi"}],
            model="glm-5.3-flash",
            role="weak",
            reasoning_effort="low",
        )
    )
    assert outcome.ok
    assert captured["json"]["reasoning_effort"] == "low"
    plain = GenerationClient(base_url="http://x", api_key="k")
    plain._client = _Client()
    asyncio.run(plain.generate([{"role": "user", "content": "hi"}], model="m", role="weak"))
    assert "reasoning_effort" not in captured["json"]
