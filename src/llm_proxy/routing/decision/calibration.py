"""Platt/temperature scaling for ensemble confidence calibration.

Ports the upstream UncommonRoute calibrator (pure stdlib, no dependencies).
The vendored temperature (``assets/calibration_params.json``) is applied to
the ensemble's raw confidence so downstream thresholds (direct/conservative,
LOW-tier escalation) operate on calibrated probabilities.

``fit_platt_from_evals`` has no production caller yet; it is kept for future
online re-fitting once the explicit feedback loop has accumulated enough
calibration samples, and is covered by unit tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from llm_proxy.observability.logger import get_logger
from llm_proxy.routing.assets import asset_path

logger = get_logger(__name__)


@dataclass
class PlattCalibrator:
    """Temperature-scaling calibrator matching the ``_Calibrator`` protocol."""

    temperature: float = 1.0

    def calibrate(self, raw: float) -> float:
        safe = max(1e-6, min(1.0 - 1e-6, raw))
        logit = math.log(safe / (1.0 - safe))
        scaled = logit / max(0.05, self.temperature)
        return 1.0 / (1.0 + math.exp(-scaled))


def load_calibration_temperature() -> float:
    """Load the vendored Platt temperature; fall back to 1.0 on any failure.

    Degrades gracefully (identity calibration) instead of crashing, matching
    the asset-degradation policy used elsewhere in the routing package.
    """
    try:
        path = asset_path("calibration_params.json")
        return float(json.loads(path.read_text())["temperature"])
    except Exception:
        logger.warning("calibration_params.json unavailable; using temperature 1.0", exc_info=True)
        return 1.0


def save_calibrator(calibrator: PlattCalibrator, path: Path) -> None:
    """Persist the temperature as JSON (used by future online re-fitting)."""
    path.write_text(json.dumps({"temperature": calibrator.temperature, "version": 1}))


def load_calibrator(path: Path) -> PlattCalibrator:
    """Load a calibrator saved by ``save_calibrator`` (tests / future tooling)."""
    data = json.loads(path.read_text())
    return PlattCalibrator(temperature=float(data["temperature"]))


def fit_platt_from_evals(
    evals: list[dict],
    min_temperature: float = 0.5,
    max_temperature: float = 3.0,
    step: float = 0.05,
) -> PlattCalibrator:
    """Grid-search the temperature minimizing ECE over recorded eval samples.

    Each eval item must carry ``confidence`` (raw ensemble confidence) and
    ``correct`` (whether the routed tier proved adequate, e.g. derived from
    explicit feedback).
    """
    best_temp = 1.0
    best_ece = float("inf")
    temp = min_temperature
    while temp <= max_temperature + 1e-9:
        ece = _compute_ece(evals, temp)
        if ece < best_ece:
            best_ece = ece
            best_temp = round(temp, 4)
        temp += step
    return PlattCalibrator(temperature=best_temp)


def _compute_ece(evals: list[dict], temperature: float, buckets: int = 10) -> float:
    """Expected calibration error of a temperature over eval samples."""
    if not evals:
        return 0.0
    cal = PlattCalibrator(temperature=temperature)
    bucket_data: list[list[tuple[float, float]]] = [[] for _ in range(buckets)]
    for item in evals:
        conf = cal.calibrate(item["confidence"])
        correct = 1.0 if item["correct"] else 0.0
        idx = min(int(conf * buckets), buckets - 1)
        bucket_data[idx].append((conf, correct))
    ece = 0.0
    total = len(evals)
    for bucket in bucket_data:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(a for _, a in bucket) / len(bucket)
        ece += abs(avg_conf - avg_acc) * len(bucket) / total
    return ece
