import numpy as np
import pytest

from btc.gradcam import cam_from_features, gradcam
from btc.models import mobilenet_v2, small_cnn, with_features


def test_small_cnn_shapes():
    m = small_cnn()
    assert m.output_shape == (None, 1)
    assert m.count_params() < 500_000
    f = with_features(m)
    assert f.outputs[1].shape[1:] == (14, 14, 128)


def test_mobilenet_shapes_without_download():
    m = mobilenet_v2(weights=None)
    assert m.output_shape == (None, 1)
    assert with_features(m).outputs[1].shape[1:] == (7, 7, 1280)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_closed_form_cam_equals_autodiff_gradcam(sign):
    m = small_cnn()
    f = with_features(m)
    x = np.random.default_rng(0).uniform(0, 255, (224, 224, 3)).astype("float32")
    auto = gradcam(f, x, sign=sign)
    _, fmap = f(x[None], training=False)
    closed = cam_from_features(fmap.numpy()[0], m.get_layer("logits").get_weights()[0], sign=sign)
    np.testing.assert_allclose(auto, closed, rtol=1e-4, atol=1e-6)
