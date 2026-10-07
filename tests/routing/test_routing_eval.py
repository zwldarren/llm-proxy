"""Tests for the offline routing evaluation rig (ADR-0018).

The rig is what makes a routing change falsifiable, so it is tested like
production code: corpus validation, the ensemble leg, the metric split between
under- and over-tiering, judge abstention handling, and the CLI including the
judge leg.
"""

import json

import pytest

from llm_proxy.routing.eval import (
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_REPEATS,
    DEFAULT_CASES_PATH,
    CaseResult,
    EvalCase,
    bootstrap_match_rate_ci,
    load_cases,
    render_report,
    run_ensemble_leg,
    summarize,
)
from llm_proxy.routing.eval.judges import (
    OPENROUTER_JUDGE_URL,
    JudgeOutcome,
    OllamaSystemOneJudge,
    OpenRouterSystemOneJudge,
)
from llm_proxy.routing.judge import JudgeVerdict
from llm_proxy.routing.types import Tier


def _case(
    case_id: str = "case-1",
    *,
    subset: str = "short",
    tier: Tier = Tier.MEDIUM,
    content: str = "Summarize this in one sentence.",
) -> EvalCase:
    return EvalCase(
        id=case_id,
        subset=subset,
        messages=[{"role": "user", "content": content}],
        expected_tier=tier,
        rationale="authored for the test",
    )


def _case_line(**overrides: object) -> str:
    """One valid corpus line, with fields overridden for the invalid cases."""
    payload: dict[str, object] = {
        "id": "case-1",
        "subset": "short",
        "expected_tier": "SIMPLE",
        "rationale": "authored for the test",
        "messages": [{"role": "user", "content": "x"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _result(
    *,
    expected: Tier,
    ensemble: Tier,
    confidence: float = 0.9,
    complexity: float = 0.5,
    judge: Tier | None = None,
    abstained: bool = False,
    latency_ms: float | None = None,
) -> CaseResult:
    return CaseResult(
        case=_case(tier=expected),
        repeat=0,
        ensemble_tier=ensemble,
        ensemble_complexity=complexity,
        ensemble_confidence=confidence,
        judge_tier=judge,
        judge_abstained=abstained,
        judge_called=judge is not None or abstained,
        judge_latency_ms=latency_ms if latency_ms is not None else (5.0 if judge else None),
    )


class TestCorpus:
    def test_packaged_corpus_is_well_formed(self):
        cases = load_cases()
        assert len(cases) >= 8
        assert len({case.id for case in cases}) == len(cases)
        assert all(case.rationale for case in cases), "every label needs an authored reason"
        assert {case.expected_tier for case in cases} == set(Tier)

    def test_explicit_path_is_loaded(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "only",
                    "subset": "boundary",
                    "expected_tier": "COMPLEX",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ),
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert [case.id for case in cases] == ["only"]
        assert cases[0].expected_tier is Tier.COMPLEX
        # The loader tolerates a missing rationale (the README asks for one, the
        # schema does not), so a hand-written corpus file stays easy to author.
        assert cases[0].rationale == ""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"expected_tier": "EASY"},
            {"subset": "unknown"},
            {"id": ""},
            {"messages": []},
            {"messages": "not-a-list"},
        ],
    )
    def test_invalid_corpus_lines_are_rejected(self, tmp_path, overrides):
        path = tmp_path / "cases.jsonl"
        path.write_text(_case_line(**overrides) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cases.jsonl:1"):
            load_cases(path)

    def test_malformed_json_is_rejected(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cases.jsonl:1"):
            load_cases(path)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        line = _case_line(id="dup")
        path = tmp_path / "cases.jsonl"
        path.write_text(f"{line}\n{line}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_cases(path)

    def test_empty_corpus_is_rejected(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_cases(path)

    def test_packaged_corpus_path_exists(self):
        assert DEFAULT_CASES_PATH.exists()


class TestEnsembleLeg:
    def test_ensemble_leg_is_deterministic_for_a_seed(self):
        import random

        case = _case(content="Write a haiku about the sea.")
        first = run_ensemble_leg(case, rng=random.Random("seed:case"))
        second = run_ensemble_leg(case, rng=random.Random("seed:case"))
        assert first == second
        tier, complexity, confidence = first
        assert isinstance(tier, Tier)
        assert 0.0 <= complexity <= 1.0
        assert 0.0 <= confidence <= 1.0


class TestSummarize:
    def test_under_and_over_are_reported_separately(self):
        metrics = summarize(
            [
                _result(expected=Tier.COMPLEX, ensemble=Tier.SIMPLE),
                _result(expected=Tier.SIMPLE, ensemble=Tier.COMPLEX),
            ]
        )
        overall = metrics["ensemble"]["overall"]
        assert overall["under"] == 1
        assert overall["over"] == 1
        assert overall["match_rate"] == 0.0

    def test_judge_abstention_is_excluded_from_the_match_rate(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX,
                    ensemble=Tier.COMPLEX,
                    judge=None,
                    abstained=True,
                )
            ]
        )
        verdict_only = metrics["judge"]["verdict_only"]["overall"]
        assert verdict_only["abstained"] == 1
        assert verdict_only["decided"] == 0
        assert verdict_only["match_rate"] is None

    def test_gate_call_rate_counts_paid_calls_even_when_the_judge_abstains(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX,
                    ensemble=Tier.MEDIUM,
                    confidence=0.30,
                    judge=None,
                    abstained=True,
                )
            ]
        )
        gate = metrics["judge"]["gates"]["confidence"]
        # The gate fired and the judge was billed, even though the verdict was
        # unusable: the paid call rate must not depend on the verdict's usefulness.
        assert gate["calls"] == 1
        assert gate["rate"] == 1.0

    def test_confidence_gate_calls_the_judge_only_below_the_threshold(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX, ensemble=Tier.MEDIUM, confidence=0.90, judge=Tier.COMPLEX
                ),
                _result(
                    expected=Tier.COMPLEX, ensemble=Tier.MEDIUM, confidence=0.30, judge=Tier.COMPLEX
                ),
            ]
        )
        assert metrics["judge"]["always"]["overall"]["match_rate"] == 1.0
        gate = metrics["judge"]["gates"]["confidence"]
        assert gate["calls"] == 1
        assert gate["overall"]["matched"] == 1
        assert gate["overall"]["under"] == 1

    def test_gate_sweep_reports_every_candidate_trigger(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX, ensemble=Tier.MEDIUM, confidence=0.30, judge=Tier.COMPLEX
                ),
            ]
        )
        gates = metrics["judge"]["gates"]
        assert {"confidence", "band", "not_low_anchor", "always"} <= set(gates)
        assert gates["always"]["calls"] == 1
        assert gates["always"]["rate"] == 1.0

    def test_quantile_gate_derives_its_threshold_from_the_corpus(self):
        results = [
            _result(
                expected=Tier.SIMPLE,
                ensemble=Tier.SIMPLE,
                confidence=value,
                judge=Tier.SIMPLE,
            )
            for value in (0.10, 0.20, 0.30, 0.40)
        ]
        gate = summarize(results)["judge"]["gates"]["confidence_q50"]
        assert gate["detail"]["quantile"] == 0.50
        assert gate["detail"]["threshold"] == pytest.approx(0.25)
        assert gate["calls"] == 2

    def test_band_gate_fires_on_the_middle_band_whatever_the_confidence(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX,
                    ensemble=Tier.MEDIUM,
                    confidence=0.95,
                    complexity=0.40,
                    judge=Tier.COMPLEX,
                ),
                _result(
                    expected=Tier.COMPLEX,
                    ensemble=Tier.MEDIUM,
                    confidence=0.95,
                    complexity=0.68,
                    judge=Tier.SIMPLE,
                ),
            ]
        )
        assert metrics["judge"]["gates"]["confidence"]["calls"] == 0
        assert metrics["judge"]["gates"]["band"]["calls"] == 1
        assert metrics["judge"]["gates"]["band"]["overall"]["match_rate"] == 0.5

    def test_confidence_distribution_is_reported(self):
        metrics = summarize(
            [
                _result(expected=Tier.SIMPLE, ensemble=Tier.SIMPLE, confidence=0.20),
                _result(expected=Tier.SIMPLE, ensemble=Tier.SIMPLE, confidence=0.90),
            ]
        )
        confidence = metrics["ensemble"]["confidence"]
        assert confidence["min"] == 0.20
        assert confidence["max"] == 0.90
        assert confidence["below_gate"] == 1

    def test_report_renders_every_path(self):
        metrics = summarize(
            [
                _result(
                    expected=Tier.COMPLEX, ensemble=Tier.MEDIUM, confidence=0.30, judge=Tier.COMPLEX
                ),
            ]
        )
        report = render_report(metrics)
        assert "# Routing evaluation report" in report
        assert "## Ensemble baseline" in report
        assert "## Judge" in report
        assert "| gate | calls | call rate | match | 95% CI | under | over |" in report
        assert "| ensemble (no judge) |" in report
        assert "| judge, every case |" in report
        assert "Ensemble confidence min/median/max" in report


class TestBootstrapIntervals:
    """The ADR's protocol: percentile intervals over whole cases, seeded."""

    def test_the_interval_is_seeded_and_deterministic(self):
        rows = [("a", 1.0), ("a", 1.0), ("b", 1.0), ("b", 1.0), ("c", 1.0), ("c", 1.0), ("d", 0.0)]
        first = bootstrap_match_rate_ci(rows)
        second = bootstrap_match_rate_ci(rows)
        assert first == second
        assert first["repeats"] == BOOTSTRAP_REPEATS
        assert first["level"] == BOOTSTRAP_LEVEL

    def test_the_interval_brackets_the_point_estimate(self):
        rows = [("a", 1.0), ("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 0.0)]
        ci = bootstrap_match_rate_ci(rows, repeats=2000)
        point = sum(flag for _, flag in rows) / len(rows)
        assert ci["low"] <= point <= ci["high"]

    def test_whole_cases_are_the_unit_of_resampling(self):
        """One case with disagreeing repeats is drawn whole, never split.

        Case ``a`` is matched on one repeat and mismatched on the other; every
        draw that includes ``a`` carries both rows, so the replicate rate is
        always 0.5 and the interval collapses to it — it can never reach 1.0.
        """
        ci = bootstrap_match_rate_ci([("a", 1.0), ("a", 0.0)], repeats=500)
        assert ci["low"] == 0.5
        assert ci["high"] == 0.5

    def test_abstentions_are_excluded_not_mismatches(self):
        assert bootstrap_match_rate_ci([("a", None), ("b", None)]) is None
        ci = bootstrap_match_rate_ci([("a", None), ("b", 1.0)])
        assert ci["low"] == 1.0 and ci["high"] == 1.0

    def test_zero_repeats_disable_intervals(self):
        assert bootstrap_match_rate_ci([("a", 1.0)], repeats=0) is None

    def test_summarize_attaches_intervals_and_zero_disables_them(self):
        results = [_result(expected=Tier.COMPLEX, ensemble=Tier.MEDIUM, judge=Tier.COMPLEX)]
        metrics = summarize(results)
        assert metrics["ensemble"]["overall"]["match_rate_ci"] is not None
        assert metrics["judge"]["always"]["overall"]["match_rate_ci"] is not None
        assert metrics["judge"]["verdict_only"]["overall"]["match_rate_ci"] is not None
        assert metrics["judge"]["gates"]["band"]["match_rate_ci"] is not None
        off = summarize(results, bootstrap=0)
        assert off["ensemble"]["overall"]["match_rate_ci"] is None
        assert off["judge"]["gates"]["always"]["match_rate_ci"] is None


class TestOllamaTransport:
    async def test_response_is_parsed_into_a_verdict(self, mock_response_cls, make_mock_client):
        judge = OllamaSystemOneJudge(model="tev1:0.8b")
        judge._client = make_mock_client(
            mock_response_cls(
                json_data={
                    "model": "tev1:0.8b",
                    "answers": {
                        "tier": {
                            "type": "choice",
                            "choice": "medium",
                            "probabilities": {"medium": 0.9},
                        },
                        "escalate": {"type": "noul", "noul": 0.2},
                    },
                    "usage": {"input_tokens": 120, "output_tokens": 8},
                }
            )
        )
        outcome = await judge.evaluate({"current_request": "hi"})
        await judge.aclose()
        assert outcome.error is None
        assert outcome.verdict is not None and outcome.verdict.tier is Tier.MEDIUM
        assert outcome.verdict.escalation_probability == 0.2
        assert outcome.judge_model == "tev1:0.8b"
        assert outcome.input_tokens == 120
        assert outcome.latency_ms >= 0.0
        assert judge._client is None

    async def test_failure_is_an_abstention_not_an_exception(self, make_mock_client):
        judge = OllamaSystemOneJudge()
        client = make_mock_client()
        client.post.side_effect = RuntimeError("connection refused")
        judge._client = client
        outcome = await judge.evaluate({"current_request": "hi"})
        assert outcome.verdict is None
        assert outcome.abstained is True
        assert "connection refused" in (outcome.error or "")


class TestOpenRouterTransport:
    """Jev is the reference judge the ADR compares against, so the rig can reach it."""

    ANSWER = {
        "model": "~typesafe/jev-latest",
        "answers": {
            "tier": {"type": "choice", "choice": "complex", "probabilities": {"complex": 1.0}},
            "escalate": {"type": "noul", "noul": 0.7},
        },
        "usage": {"input_tokens": 983, "output_tokens": 66},
    }

    async def test_the_call_goes_to_systemone_with_a_bearer_token(
        self, monkeypatch, mock_response_cls, make_mock_client
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        judge = OpenRouterSystemOneJudge()
        client = make_mock_client(mock_response_cls(json_data=self.ANSWER))
        judge._client = client

        outcome = await judge.evaluate({"current_request": "hi"})
        await judge.aclose()

        assert outcome.verdict is not None and outcome.verdict.tier is Tier.COMPLEX
        assert outcome.input_tokens == 983
        assert client.post.await_args.args[0] == f"{OPENROUTER_JUDGE_URL}/systemone"
        assert client.post.await_args.kwargs["headers"] == {"Authorization": "Bearer sk-or-env"}
        assert client.post.await_args.kwargs["json"]["model"] == "~typesafe/jev-latest"
        assert judge._client is None

    def test_a_missing_credential_is_a_usage_error_not_a_silent_abstention(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterSystemOneJudge()

    def test_the_cli_reports_a_missing_credential(self, monkeypatch, capsys):
        import llm_proxy.routing.eval.__main__ as cli

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        assert cli.main(["--judge", "openrouter"]) == 2
        assert "OPENROUTER_API_KEY" in capsys.readouterr().err


class _StubJudge:
    """A judge that always answers COMPLEX, for the CLI test."""

    name = "stub"
    model = "stub-judge"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def evaluate(self, state, questions=None):  # noqa: ANN001, ANN201 - test double
        self.calls += 1
        return JudgeOutcome(
            verdict=JudgeVerdict(tier=Tier.COMPLEX, probabilities={"complex": 0.9}),
            latency_ms=5.0,
            judge_model="stub-judge",
        )

    async def aclose(self) -> None:
        self.closed = True


class TestCli:
    def test_cli_runs_the_corpus_with_a_judge_and_writes_artifacts(
        self, tmp_path, monkeypatch, capsys
    ):
        corpus = tmp_path / "cases.jsonl"
        corpus.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"case-{i}",
                        "subset": "boundary",
                        "expected_tier": "COMPLEX",
                        "rationale": "authored for the test",
                        "messages": [{"role": "user", "content": f"do the hard thing {i}"}],
                    }
                )
                for i in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        stub = _StubJudge()
        import llm_proxy.routing.eval.__main__ as cli

        monkeypatch.setattr(cli, "_build_judge", lambda args: stub)

        out = tmp_path / "out"
        assert cli.main(["--cases", str(corpus), "--judge", "ollama", "--out", str(out)]) == 0

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        # Default protocol: three seeded repeats (n = cases × repeats).
        assert metrics["n"] == 6
        assert metrics["run"]["repeats"] == 3
        assert metrics["run"]["judge"] == "stub"
        assert metrics["run"]["bootstrap"] == BOOTSTRAP_REPEATS
        assert metrics["judge"]["always"]["overall"]["matched"] == 6
        assert stub.calls == 6
        assert stub.closed is True
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "## Judge" in report
        assert "# Routing evaluation report" in capsys.readouterr().out

    def test_a_non_positive_limit_is_a_usage_error(self, capsys):
        import llm_proxy.routing.eval.__main__ as cli

        with pytest.raises(SystemExit) as excinfo:
            cli._parse_args(["--limit", "0"])

        assert excinfo.value.code == 2
        assert "positive integer" in capsys.readouterr().err

    def test_a_missing_corpus_is_a_usage_error_not_a_traceback(self, tmp_path, capsys):
        import llm_proxy.routing.eval.__main__ as cli

        missing = tmp_path / "nope.jsonl"
        assert cli.main(["--cases", str(missing)]) == 2
        assert "nope.jsonl" in capsys.readouterr().err
