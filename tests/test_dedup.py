import io

import numpy as np
from PIL import Image, ImageDraw

from btc.data import connected_components, deduplicate, hamming_matrix, phash_bits


def synthetic_scan(seed: int, size: int = 256) -> Image.Image:
    """A grey 'head' ellipse with a few random blobs, enough structure for pHash."""
    rng = np.random.default_rng(seed)
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([20, 30, size - 20, size - 30], fill=90)
    for _ in range(6):
        x, y = rng.integers(40, size - 60, 2)
        r = int(rng.integers(8, 40))
        d.ellipse([x, y, x + r, y + r], fill=int(rng.integers(120, 255)))
    return im.convert("RGB")


def jpeg_roundtrip(im: Image.Image, quality: int = 70, resize: int | None = None) -> Image.Image:
    if resize:
        im = im.resize((resize, resize), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_hamming_matrix_matches_bitwise_count():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (5, 64)).astype(bool)
    d = hamming_matrix(bits)
    for i in range(5):
        for j in range(5):
            assert d[i, j] == (bits[i] != bits[j]).sum()


def test_resized_recompressed_copy_is_near_duplicate():
    a = synthetic_scan(1)
    copy = jpeg_roundtrip(a, quality=60, resize=180)
    other = synthetic_scan(2)
    d = hamming_matrix(np.array([phash_bits(a), phash_bits(copy), phash_bits(other)]))
    assert d[0, 1] <= 2
    assert d[0, 2] > 10


def test_connected_components_chains_transitively():
    # 0-1 and 1-2 are close, 3 is far from all
    dist = np.array([[0, 2, 4, 20], [2, 0, 2, 20], [4, 2, 0, 20], [20, 20, 20, 0]])
    comp = connected_components(dist, threshold=2)
    assert comp[0] == comp[1] == comp[2]
    assert comp[3] != comp[0]


def test_deduplicate_keeps_largest_and_drops_conflicts():
    imgs = [synthetic_scan(1), jpeg_roundtrip(synthetic_scan(1), 90), synthetic_scan(2),
            synthetic_scan(3), jpeg_roundtrip(synthetic_scan(3), 90)]
    bits = np.array([phash_bits(i) for i in imgs])
    labels = ["glioma", "glioma", "no_tumor", "glioma", "no_tumor"]  # 3 and 4 disagree
    sizes = [100, 200, 100, 100, 100]
    md5 = ["a", "b", "c", "d", "d"]
    res = deduplicate(labels, bits, md5, sizes, threshold=2)
    assert res.keep == [1, 2]  # larger copy of the first pair, the unique image
    assert res.removed_duplicates == [0]
    assert res.removed_conflicts == [3, 4]
    assert res.conflicting_clusters == 1
    assert res.exact_duplicate_files == 1
