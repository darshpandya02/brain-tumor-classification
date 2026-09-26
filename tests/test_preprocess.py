import numpy as np
from PIL import Image

from btc.data import load_rgb, preprocess_file, resize, to_uint8


def test_resize_is_identity_at_model_size():
    x = np.random.default_rng(0).integers(0, 256, (224, 224, 3)).astype(np.uint8)
    np.testing.assert_array_equal(resize(x), x.astype(np.float32))


def test_resize_uses_half_pixel_centers():
    # 2x downsample with half-pixel centers averages each 2x2 block exactly.
    x = np.arange(448 * 448 * 3, dtype=np.float32).reshape(448, 448, 3) % 251
    y = resize(x.astype(np.uint8))
    expected = x.reshape(224, 2, 224, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(y, expected, atol=1e-3)


def test_preprocess_file_converts_grayscale_to_rgb(tmp_path):
    g = (np.random.default_rng(1).uniform(0, 255, (300, 260))).astype(np.uint8)
    p = tmp_path / "g.png"
    Image.fromarray(g, "L").save(p)
    assert load_rgb(p).shape == (300, 260, 3)
    x = preprocess_file(p)
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert to_uint8(np.array([-3.0, 12.4, 300.0])).tolist() == [0, 12, 255]
