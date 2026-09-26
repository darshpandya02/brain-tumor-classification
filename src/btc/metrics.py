"""Evaluation metrics for the binary task, calibration, and bootstrap intervals."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, roc_auc_score


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """y: 1 = tumor. p: predicted probability of tumor."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float((tp + tn) / len(y)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": expected_calibration_error(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        # rows = true (healthy, tumor), cols = predicted (healthy, tumor)
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins on the predicted tumor probability."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": int(m.sum()),
                "mean_p": float(p[m].mean()) if m.any() else None,
                "frac_tumor": float(y[m].mean()) if m.any() else None,
            }
        )
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimizing the binary NLL of sigmoid(logits / T) (fit on validation)."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(log_t: float) -> float:
        q = np.clip(sigmoid(z / np.exp(log_t)), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float = 0.5, n: int = 2000, seed: int = 0) -> dict:
    """Percentile 95% intervals, resampling test images with replacement (stratified by class)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    keys = ["accuracy", "sensitivity", "specificity", "balanced_accuracy", "roc_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = binary_metrics(y[i], p[i], threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def multiclass_metrics(y: np.ndarray, prob: np.ndarray, class_names: list[str]) -> dict:
    from sklearn.metrics import f1_score

    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
    recall = cm.diagonal() / cm.sum(1)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_recall": {c: float(r) for c, r in zip(class_names, recall)},
        "roc_auc_ovr_macro": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
    }
