import numpy as np
import pytest

from btc.metrics import (
    binary_metrics,
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    multiclass_metrics,
    sigmoid,
)


def test_binary_metrics_known_confusion():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.3, 0.9, 0.8, 0.6, 0.4, 0.95, 0.55])
    m = binary_metrics(y, p)
    assert m["confusion_matrix"] == [[3, 1], [1, 5]]
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["sensitivity"] == pytest.approx(5 / 6)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 3 / 4) / 2)
    assert 0 <= m["roc_auc"] <= 1


def test_ece_is_small_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    # Overconfident predictions are penalised.
    assert expected_calibration_error(y, np.where(p > 0.5, 0.99, 0.01)) > 0.2


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 2, 50_000)
    y = (rng.uniform(size=z_true.size) < sigmoid(z_true)).astype(int)
    t = fit_temperature(3.0 * z_true, y)  # logits 3x too confident
    assert t == pytest.approx(3.0, rel=0.05)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.uniform(0, 0.5, 300), 0, 1)
    point = binary_metrics(y, p)
    ci = bootstrap_ci(y, p, n=200)
    for k in ("accuracy", "sensitivity", "specificity", "roc_auc"):
        assert ci[k][0] <= point[k] <= ci[k][1]


def test_multiclass_metrics():
    y = np.array([0, 1, 2, 3, 0, 1])
    prob = np.eye(4)[[0, 1, 2, 3, 1, 1]]
    prob = (prob + 0.01) / (prob + 0.01).sum(1, keepdims=True)
    m = multiclass_metrics(y, prob, ["a", "b", "c", "d"])
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["per_class_recall"]["a"] == pytest.approx(0.5)
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
