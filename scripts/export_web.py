"""Export the binary models for the browser demo and write parity fixtures.

For each binary model this writes artifacts/<name>/saved_model with one
signature returning
  logit: (B,)        raw tumor logit
  cam:   (B, h, w)   sum_k w_k A_k / (h w), the signed Grad-CAM map for the tumor logit
plus web/models/<name>/meta.json (temperature, threshold, test metrics).

It also picks the bundled sample images from the held-out test split and
records the Python outputs for them in web-tests/reference.json, which the
Node and browser parity tests compare against.

Run scripts/convert_tfjs.sh afterwards to produce the TF.js graph models.
"""

from __future__ import annotations

import json
import shutil

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from btc import config
from btc.data import load_rgb, load_split_arrays, preprocess_file, read_manifest
from btc.metrics import sigmoid
from btc.models import FEATURES_LAYER, with_features

MODELS = {"cnn_binary": "Small CNN (from scratch)", "mobilenet_binary": "MobileNetV2 (transfer learning)"}
SAMPLES_PER_CLASS = {"no_tumor": 6, "glioma_tumor": 2, "meningioma_tumor": 2, "pituitary_tumor": 2}
UPLOAD_FIXTURES = 2  # extra test-split JPEGs (not bundled) used to test the upload path


class Serving(tf.Module):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.features_model = with_features(model)
        self.w = tf.constant(model.get_layer("logits").get_weights()[0][:, 0], tf.float32)
        _, h, w, _ = self.features_model.outputs[1].shape
        self.hw = float(h * w)

    @tf.function(input_signature=[tf.TensorSpec([None, config.IMG_SIZE, config.IMG_SIZE, 3], tf.float32, name="image")])
    def serve(self, image):
        logits, fmap = self.features_model(image, training=False)
        # sum_k w_k A_k as reshape + matmul (TF.js WASM has no Einsum kernel).
        _, h, w, k = self.features_model.outputs[1].shape
        cam = tf.reshape(tf.matmul(tf.reshape(fmap, [-1, k]), self.w[:, None]), [-1, h, w]) / self.hw
        return {"logit": logits[:, 0], "cam": cam}


def pick_samples(rows):
    rng = np.random.default_rng(config.SEED)
    test = [r for r in rows if r["split"] == "test"]
    picked = []
    for label, n in SAMPLES_PER_CLASS.items():
        pool = sorted((r for r in test if r["label4"] == label), key=lambda r: r["path"])
        picked += [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
    rest = sorted((r for r in test if r not in picked), key=lambda r: r["path"])
    uploads = [rest[i] for i in sorted(rng.choice(len(rest), UPLOAD_FIXTURES, replace=False))]
    return picked, uploads


def main() -> None:
    rows = read_manifest()
    x_test, _, _, test_rows = load_split_arrays("test", rows)
    by_path = {r["path"]: i for i, r in enumerate(test_rows)}
    samples, uploads = pick_samples(rows)

    sample_dir = config.WEB_DIR / "samples"
    if sample_dir.exists():
        for p in sample_dir.glob("*.png"):
            p.unlink()
    sample_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = config.ROOT / "web-tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    sample_meta, sample_x = [], []
    for k, r in enumerate(samples):
        img = x_test[by_path[r["path"]]]  # uint8, already resized exactly as in evaluation
        fname = f"sample_{k:02d}.png"
        Image.fromarray(img).save(sample_dir / fname)
        sample_x.append(img.astype(np.float32))
        sample_meta.append(
            {
                "file": fname,
                "label": "tumor" if r["tumor"] else "healthy",
                "tumor_type": r["label4"].replace("_tumor", "") if r["tumor"] else None,
                "source": f"{config.HF_REPO}/{r['path']}",
            }
        )
    upload_meta, upload_x = [], []
    for k, r in enumerate(uploads):
        # Each upload fixture exists as the original JPEG and as a lossless PNG at
        # the original resolution. The PNG decodes to identical pixels everywhere,
        # so it tests the resize path exactly; the JPEG tests a realistic upload
        # (browser and PIL JPEG decoders may differ by a few grey levels).
        label = "tumor" if r["tumor"] else "healthy"
        jpg, png = f"upload_{k}.jpg", f"upload_{k}.png"
        shutil.copyfile(config.RAW_DIR / r["path"], fixture_dir / jpg)
        Image.fromarray(load_rgb(fixture_dir / jpg)).save(fixture_dir / png)
        for fname in (png, jpg):
            upload_x.append(preprocess_file(fixture_dir / fname))
            upload_meta.append({"file": fname, "label": label, "lossless": fname.endswith(".png")})

    reference = {"samples": sample_meta, "uploads": upload_meta, "models": {}}
    manifest = {"models": []}
    for name, title in MODELS.items():
        model = keras.models.load_model(config.ARTIFACTS_DIR / name / "model.keras")
        report = json.loads((config.REPORTS_DIR / f"{name}.json").read_text())
        module = Serving(model)
        sm_dir = config.ARTIFACTS_DIR / name / "saved_model"
        if sm_dir.exists():
            shutil.rmtree(sm_dir)
        tf.saved_model.save(module, str(sm_dir), signatures={"serving_default": module.serve})

        t = report["temperature"]
        ref = {}
        for key, xs in (("samples", sample_x), ("uploads", upload_x)):
            out = module.serve(tf.constant(np.stack(xs)))
            logit = out["logit"].numpy()
            ref[key] = [
                {"logit": float(z), "prob": float(sigmoid(z / t)), "cam": np.round(c, 6).tolist()}
                for z, c in zip(logit, out["cam"].numpy())
            ]
        reference["models"][name] = ref

        # Sanity: the exported logit equals the Keras model's logit.
        direct = model.predict(np.stack(sample_x), verbose=0)[:, 0]
        assert np.allclose(direct, [r["logit"] for r in ref["samples"]], atol=1e-4)

        test = report["test"]
        meta = {
            "name": name,
            "title": title,
            "input_size": config.IMG_SIZE,
            "input_range": [0, 255],
            "positive_class": "tumor",
            "temperature": t,
            "threshold": 0.5,
            "features_layer": FEATURES_LAYER,
            "params": report["params"],
            "test_metrics": {
                k: test[k]
                for k in ("n", "accuracy", "sensitivity", "specificity", "roc_auc", "ece", "brier", "confusion_matrix")
            },
            "test_ci95": report["test_ci95"],
        }
        mdir = config.WEB_DIR / "models" / name
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest["models"].append({"name": name, "title": title})
        print(name, "exported; sample probs", [round(r["prob"], 3) for r in ref["samples"]])

    (sample_dir / "samples.json").write_text(json.dumps(sample_meta, indent=2) + "\n")
    (config.WEB_DIR / "models" / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (config.ROOT / "web-tests" / "reference.json").write_text(json.dumps(reference) + "\n")


if __name__ == "__main__":
    main()
