"""Model definitions.

Both models take float32 RGB images in [0, 255] of shape (224, 224, 3) and do
their own input scaling, so the browser can feed raw pixel values. Both end in
GlobalAveragePooling -> Dropout -> Dense, which makes Grad-CAM on the last
convolutional feature map computable from the dense weights alone (see gradcam.py).
"""

from __future__ import annotations

import keras
from keras import layers

from . import config

FEATURES_LAYER = "features"  # name of the last conv feature map in both models


def small_cnn(num_outputs: int = 1, size: int = config.IMG_SIZE, dropout: float = 0.3) -> keras.Model:
    """A small CNN trained from scratch (about 0.3M parameters, 224 -> 14x14x128 features)."""
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0, name="scale")(inp)

    def block(x, filters, i, stride=1, last=False):
        x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        return layers.ReLU(name=FEATURES_LAYER if last else f"relu{i}")(x)

    x = block(x, 32, 0, stride=2)  # 112
    x = block(x, 32, 1)
    x = layers.MaxPooling2D(2, name="pool1")(x)  # 56
    x = block(x, 64, 2)
    x = layers.MaxPooling2D(2, name="pool2")(x)  # 28
    x = block(x, 128, 3)
    x = layers.MaxPooling2D(2, name="pool3")(x)  # 14
    x = block(x, 128, 4)
    x = block(x, 128, 5, last=True)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_outputs, name="logits")(x)
    return keras.Model(inp, out, name="small_cnn")


def mobilenet_v2(
    num_outputs: int = 1,
    size: int = config.IMG_SIZE,
    dropout: float = 0.3,
    weights: str | None = "imagenet",
) -> keras.Model:
    """MobileNetV2 (ImageNet weights) with a new linear head."""
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale")(inp)
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=weights, name="backbone"
    )
    # training=False keeps BatchNorm statistics frozen during fine-tuning.
    x = base(x, training=False)
    x = layers.Identity(name=FEATURES_LAYER)(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(num_outputs, name="logits")(x)
    return keras.Model(inp, out, name="mobilenet_v2")


def with_features(model: keras.Model) -> keras.Model:
    """Model returning (logits, last conv feature map) for Grad-CAM and export."""
    feats = model.get_layer(FEATURES_LAYER).output
    return keras.Model(model.inputs, [model.output, feats], name=model.name + "_with_features")
