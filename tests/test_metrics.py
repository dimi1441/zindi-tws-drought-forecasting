"""Vérifie le calcul des métriques et leur agrégation moyenne ± écart-type across folds."""

import numpy as np
import pytest

from src.validation.metrics import aggregate_fold_metrics, compute_metrics


def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_known_mae():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.0)


def test_aggregate_fold_metrics_mean_and_std():
    fold_metrics = [{"mae": 1.0}, {"mae": 3.0}]
    aggregated = aggregate_fold_metrics(fold_metrics)
    assert aggregated["mae_mean"] == pytest.approx(2.0)
    assert aggregated["mae_std"] == pytest.approx(1.0)
