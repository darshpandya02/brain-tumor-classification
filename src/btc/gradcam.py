"""Grad-CAM.

For a head of the form logits = Dense(Dropout(GAP(A))), the gradient of logit c
with respect to the feature map A_k is w_kc / (H * W) at every location, so the
Grad-CAM channel weights are exactly w_kc / (H * W) and Grad-CAM reduces to
ReLU(sum_k w_kc A_k) / (H * W). `cam_from_features` implements that closed form
(it is what the browser demo computes); `gradcam` computes it with autodiff and
is used to test the closed form.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def gradcam(features_model, image: np.ndarray, class_index: int = 0, sign: float = 1.0) -> np.ndarray:
    """Autodiff Grad-CAM. `features_model` returns (logits, feature_map)."""
    x = tf.convert_to_tensor(image[None], tf.float32)
    with tf.GradientTape() as tape:
        logits, fmap = features_model(x, training=False)
        target = sign * logits[:, class_index]
    grads = tape.gradient(target, fmap)[0]  # (H, W, K)
    alpha = tf.reduce_mean(grads, axis=(0, 1))  # (K,)
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * alpha, axis=-1))
    return cam.numpy()


def cam_from_features(fmap: np.ndarray, dense_w: np.ndarray, class_index: int = 0, sign: float = 1.0) -> np.ndarray:
    """Closed-form Grad-CAM from the (H, W, K) feature map and (K, C) dense kernel."""
    h, w, _ = fmap.shape
    cam = fmap @ (sign * dense_w[:, class_index]) / (h * w)
    return np.maximum(cam, 0.0)


def normalize(cam: np.ndarray) -> np.ndarray:
    m = cam.max()
    return cam / m if m > 0 else cam
