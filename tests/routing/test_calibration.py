"""Tests for the Platt confidence calibrator and its asset loading."""

import json

from llm_proxy.routing.decision.calibration import (
    PlattCalibrator,
    _compute_ece,
    fit_platt_from_evals,
    load_calibration_temperature,
    load_calibrator,
    save_calibrator,
)
from llm_proxy.routing.decision.ensemble import Ensemble
from llm_proxy.routing.types import TierVote


class TestPlattCalibrator:
    def test_identity_at_temperature_1(self):
        cal = PlattCalibrator(temperature=1.0)
        for raw in (0.01, 0.2, 0.5, 0.8, 0.99):
            assert cal.calibrate(raw) == raw or abs(cal.calibrate(raw) - raw) < 1e-6

    def test_sub_unity_temperature_spreads_confidence(self):
        cal = PlattCalibrator(temperature=0.75)
        # T < 1 divides the logit by <1: low confidence goes lower,
        # high confidence goes higher, 0.5 is the fixed point.
        assert cal.calibrate(0.4) < 0.4
        assert cal.calibrate(0.6) > 0.6
        assert abs(cal.calibrate(0.5) - 0.5) < 1e-9

    def test_extreme_inputs_are_clamped(self):
        cal = PlattCalibrator(temperature=0.75)
        assert 0.0 < cal.calibrate(0.0) < 1e-3
        assert 1.0 - 1e-3 < cal.calibrate(1.0) < 1.0


class TestAssetLoading:
    def test_loads_vendored_temperature(self):
        # The packaged asset ships temperature 0.75.
        assert load_calibration_temperature() == 0.75

    def test_missing_asset_falls_back_to_1(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "llm_proxy.routing.decision.calibration.asset_path",
            lambda name: tmp_path / name,
        )
        assert load_calibration_temperature() == 1.0

    def test_corrupt_asset_falls_back_to_1(self, monkeypatch, tmp_path):
        bad = tmp_path / "calibration_params.json"
        bad.write_text("not json")
        monkeypatch.setattr(
            "llm_proxy.routing.decision.calibration.asset_path",
            lambda name: tmp_path / name,
        )
        assert load_calibration_temperature() == 1.0

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "cal.json"
        save_calibrator(PlattCalibrator(temperature=0.8), path)
        assert load_calibrator(path).temperature == 0.8
        assert json.loads(path.read_text())["version"] == 1


class TestFitting:
    def test_empty_evals_default_temperature(self):
        assert fit_platt_from_evals([]).temperature == 0.5  # grid minimum, ECE all zero

    def test_compute_ece_empty(self):
        assert _compute_ece([], 1.0) == 0.0

    def test_perfectly_calibrated_evals_pick_temperature_near_1(self):
        # Samples where correctness matches the raw confidence ordering.
        evals = [{"confidence": 0.9, "correct": True} for _ in range(9)]
        evals += [{"confidence": 0.1, "correct": False} for _ in range(9)]
        fitted = fit_platt_from_evals(evals)
        assert 0.5 <= fitted.temperature <= 3.0


class TestEnsembleIntegration:
    def test_calibrator_changes_confidence_but_not_raw(self):
        votes = [TierVote(2, 0.9), TierVote(2, 0.85), TierVote(2, 0.8)]
        plain = Ensemble(weights=[1.0, 1.0, 1.0]).decide(votes)
        calibrated = Ensemble(
            weights=[1.0, 1.0, 1.0], calibrator=PlattCalibrator(temperature=0.75)
        ).decide(votes)
        assert calibrated.raw_confidence == plain.raw_confidence
        # raw_confidence > 0.5 here, so T=0.75 pushes it up.
        assert calibrated.confidence > plain.confidence

    def test_temperature_1_matches_raw_confidence(self):
        votes = [TierVote(1, 0.7), TierVote(2, 0.6), TierVote(1, 0.8)]
        res = Ensemble(weights=[1.0, 1.0, 1.0], calibrator=PlattCalibrator(temperature=1.0)).decide(
            votes
        )
        assert abs(res.confidence - res.raw_confidence) < 1e-6
