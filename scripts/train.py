"""Train and evaluate one model.

    uv run python scripts/train.py --model cnn --task binary
    uv run python scripts/train.py --model mobilenet --task binary
    uv run python scripts/train.py --model mobilenet --task 4class

Writes artifacts/<name>/model.keras, artifacts/<name>/predictions.npz and
reports/<name>.json (test metrics, bootstrap intervals, calibration).
"""

from __future__ import annotations

import argparse
import json
import time

import keras
from keras import layers
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from btc import config
from btc.data import load_split_arrays, read_manifest
from btc.metrics import (
    binary_metrics,
    bootstrap_ci,
    fit_temperature,
    multiclass_metrics,
    reliability_bins,
    sigmoid,
)
from btc.models import mobilenet_v2, small_cnn


def augmenter(seed: int) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, fill_mode="constant", seed=seed),
            layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
            layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
            layers.RandomContrast(0.1, value_range=(0, 255), seed=seed),
        ],
        name="augment",
    )



def make_ds(x, y, batch, shuffle, aug=None, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch).map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if aug is not None:
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    ap.add_argument("--task", choices=["binary", "4class"], default="binary")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    name = f"{args.model}_{args.task}" + ("" if args.seed == config.SEED else f"_seed{args.seed}")
    out_dir = config.ARTIFACTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest()
    xtr, y2tr, y4tr, _ = load_split_arrays("train", rows)
    xva, y2va, y4va, _ = load_split_arrays("val", rows)
    xte, y2te, y4te, te_rows = load_split_arrays("test", rows)
    binary = args.task == "binary"
    ytr, yva, yte = (y2tr, y2va, y2te) if binary else (y4tr, y4va, y4te)
    n_out = 1 if binary else 4
    ytr_fit = ytr.astype("float32")[:, None] if binary else ytr
    yva_fit = yva.astype("float32")[:, None] if binary else yva

    classes = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("class weights", class_weight)

    aug = augmenter(args.seed)
    train_ds = make_ds(xtr, ytr_fit, args.batch, True, aug, args.seed)
    val_ds = make_ds(xva, yva_fit, args.batch, False)

    if binary:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [keras.metrics.AUC(from_logits=True, name="auc")]
        monitor = "val_auc"
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]
        monitor = "val_loss"
    mode = "max" if monitor == "val_auc" else "min"

    def callbacks(patience):
        return [
            keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

    t0 = time.time()
    history = {}
    if args.model == "cnn":
        model = small_cnn(n_out)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs or 40,
                      class_weight=class_weight, callbacks=callbacks(8), verbose=2)
        history["train"] = h.history
    else:
        model = mobilenet_v2(n_out)
        backbone = model.get_layer("backbone")
        backbone.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs or 10,
                      class_weight=class_weight, callbacks=callbacks(4), verbose=2)
        history["head"] = h.history
        # Fine-tune the top of the backbone (BatchNorm stays in inference mode).
        backbone.trainable = True
        for layer in backbone.layers[:-40]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(5e-5), loss=loss, metrics=metrics)
        h = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                      class_weight=class_weight, callbacks=callbacks(6), verbose=2)
        history["finetune"] = h.history
    train_seconds = time.time() - t0

    model.save(out_dir / "model.keras")
    lva = model.predict(xva.astype("float32"), batch_size=64, verbose=0)
    lte = model.predict(xte.astype("float32"), batch_size=64, verbose=0)
    np.savez(out_dir / "predictions.npz", val_logits=lva, val_y=yva, test_logits=lte, test_y=yte,
             test_paths=np.array([r["path"] for r in te_rows]))

    report = {
        "name": name,
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "params": int(model.count_params()),
        "train_seconds": round(train_seconds, 1),
        "epochs_run": {k: len(v["loss"]) for k, v in history.items()},
        "class_weight": class_weight,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
    }
    if binary:
        zva, zte = lva[:, 0], lte[:, 0]
        t = fit_temperature(zva, yva)
        p_raw, p_cal = sigmoid(zte), sigmoid(zte / t)
        report["temperature"] = t
        report["val"] = binary_metrics(yva, sigmoid(zva / t))
        report["test"] = binary_metrics(yte, p_cal)
        report["test_uncalibrated"] = {k: binary_metrics(yte, p_raw)[k] for k in ("ece", "brier")}
        report["test_ci95"] = bootstrap_ci(yte, p_cal)
        report["test_reliability"] = reliability_bins(yte, p_cal)
    else:
        prob = tf.nn.softmax(lte).numpy()
        report["test"] = multiclass_metrics(yte, prob, config.CLASSES_4)
        report["val"] = multiclass_metrics(yva, tf.nn.softmax(lva).numpy(), config.CLASSES_4)
    report["history"] = {k: {m: [float(x) for x in v] for m, v in h.items()} for k, h in history.items()}

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report["test"].items() if k != "confusion_matrix"}
    print(json.dumps({"name": name, "test": summary, "cm": report["test"]["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
