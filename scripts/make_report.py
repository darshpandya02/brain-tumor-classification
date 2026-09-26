"""Build reports/results.md and figures from the saved reports and predictions."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from btc import config
from btc.data import read_manifest
from btc.metrics import sigmoid

BINARY = {"cnn_binary": "Small CNN (from scratch)", "mobilenet_binary": "MobileNetV2 (transfer learning)"}
FIG = config.REPORTS_DIR / "figures"


def short(c: str) -> str:
    return "no tumor" if c == "no_tumor" else c.replace("_tumor", "")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    reps = {k: json.loads((config.REPORTS_DIR / f"{k}.json").read_text()) for k in BINARY}
    lines = [
        "| Model | Params | Accuracy | Sensitivity | Specificity | Balanced acc. | ROC-AUC | ECE (raw -> scaled) | Brier (raw -> scaled) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, title in BINARY.items():
        r = reps[k]
        t, ci, u = r["test"], r["test_ci95"], r["test_uncalibrated"]
        c = lambda m: f"{pct(t[m])} ({pct(ci[m][0])} to {pct(ci[m][1])})"  # noqa: E731
        lines.append(
            f"| {title} | {r['params'] / 1e6:.2f}M | {c('accuracy')} | {c('sensitivity')} | {c('specificity')} | "
            f"{c('balanced_accuracy')} | {t['roc_auc']:.3f} ({ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f}) | "
            f"{u['ece']:.3f} -> {t['ece']:.3f} | {u['brier']:.3f} -> {t['brier']:.3f} |"
        )
    lines += ["", "Confusion matrices (rows = true, columns = predicted, order healthy, tumor):", ""]
    for k, title in BINARY.items():
        cm = reps[k]["test"]["confusion_matrix"]
        lines.append(f"- {title}: healthy {cm[0]}, tumor {cm[1]} (TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]})")
    # Sensitivity by tumor type (which tumors are missed at threshold 0.5).
    label4 = {r["path"]: r["label4"] for r in read_manifest()}
    lines += ["", "Sensitivity by tumor type (test):", ""]
    for k, title in BINARY.items():
        pred = np.load(config.ARTIFACTS_DIR / k / "predictions.npz")
        p = sigmoid(pred["test_logits"][:, 0] / reps[k]["temperature"])
        types = np.array([label4[str(x)] for x in pred["test_paths"]])
        parts = []
        for c in ("glioma_tumor", "meningioma_tumor", "pituitary_tumor"):
            m = types == c
            parts.append(f"{c.replace('_tumor', '')} {int((p[m] >= 0.5).sum())}/{int(m.sum())}")
        lines.append(f"- {title}: " + ", ".join(parts))
    lines += ["", "Training: " + "; ".join(
        f"{BINARY[k]} {reps[k]['train_seconds'] / 60:.1f} min on CPU, epochs {reps[k]['epochs_run']}, temperature {reps[k]['temperature']:.2f}"
        for k in BINARY)]

    p4 = config.REPORTS_DIR / "mobilenet_4class.json"
    if p4.exists():
        r4 = json.loads(p4.read_text())["test"]
        lines += ["", "4-class variant (MobileNetV2, same splits):", "",
                  f"- accuracy {pct(r4['accuracy'])}, macro F1 {r4['macro_f1']:.3f}, macro one-vs-rest ROC-AUC {r4['roc_auc_ovr_macro']:.3f}",
                  "- per-class recall: " + ", ".join(f"{short(c)} {pct(v)}" for c, v in r4["per_class_recall"].items()),
                  f"- confusion matrix (rows = true, order {', '.join(short(c) for c in r4['classes'])}): {r4['confusion_matrix']}"]
    (config.REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figures: ROC, reliability (after temperature scaling), confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for k, title in BINARY.items():
        pred = np.load(config.ARTIFACTS_DIR / k / "predictions.npz")
        y, z = pred["test_y"], pred["test_logits"][:, 0]
        p = sigmoid(z / reps[k]["temperature"])
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{title} (AUC {reps[k]['test']['roc_auc']:.3f})")
        bins = [b for b in reps[k]["test_reliability"] if b["count"]]
        axes[1].plot([b["mean_p"] for b in bins], [b["frac_tumor"] for b in bins], "o-",
                     label=f"{title} (ECE {reps[k]['test']['ece']:.3f})")
    axes[0].plot([0, 1], [0, 1], ":", color="grey")
    axes[0].set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)", title="Test ROC")
    axes[1].plot([0, 1], [0, 1], ":", color="grey")
    axes[1].set(xlabel="Mean predicted P(tumor)", ylabel="Observed fraction tumor", title="Test reliability (10 bins)")
    for a in axes:
        a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_reliability.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for a, (k, title) in zip(axes, BINARY.items()):
        cm = np.array(reps[k]["test"]["confusion_matrix"])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["healthy", "tumor"], yticklabels=["healthy", "tumor"],
              xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=130)


if __name__ == "__main__":
    main()
