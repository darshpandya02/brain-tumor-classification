"""Dataset download, perceptual-hash deduplication, group-aware splitting and preprocessing."""

from __future__ import annotations

import csv
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from . import config


# ---------------------------------------------------------------- download


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path = config.RAW_DIR) -> None:
    """Download the pinned dataset revision from the Hugging Face Hub and unzip it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in config.HF_FILES.items():
        dest = raw_dir / name
        if not dest.exists() or sha256(dest) != expected:
            url = f"https://huggingface.co/datasets/{config.HF_REPO}/resolve/{config.HF_REVISION}/{name}"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        got = sha256(dest)
        if got != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {got}")
        split_dir = raw_dir / name.removesuffix(".zip")
        if not split_dir.exists():
            with zipfile.ZipFile(dest) as z:
                z.extractall(raw_dir)


@dataclass
class Record:
    path: str  # relative to RAW_DIR, e.g. "Training/glioma_tumor/gg (1).jpg"
    source_split: str  # "Training" or "Testing" (the original folders)
    label4: str

    @property
    def tumor(self) -> int:
        return int(self.label4 != config.HEALTHY)


def list_records(raw_dir: Path = config.RAW_DIR) -> list[Record]:
    recs = []
    for p in sorted(raw_dir.glob("*/*/*.jpg")):
        rel = p.relative_to(raw_dir)
        split, label = rel.parts[0], rel.parts[1]
        if label not in config.CLASSES_4:
            raise ValueError(f"unexpected class folder {label}")
        recs.append(Record(str(rel), split, label))
    return recs


# ---------------------------------------------------------------- hashing / dedup


def phash_bits(img: Image.Image) -> np.ndarray:
    """64-bit perceptual hash of the grayscale image as a flat bool array."""
    return imagehash.phash(img.convert("L")).hash.flatten()


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between rows of an (n, 64) bool array."""
    b = bits.astype(np.uint8)
    # distance = popcount(a xor b) = |a| + |b| - 2 a.b
    ones = b.sum(1).astype(np.int32)
    dot = b.astype(np.int32) @ b.T.astype(np.int32)
    return ones[:, None] + ones[None, :] - 2 * dot


def connected_components(dist: np.ndarray, threshold: int) -> np.ndarray:
    """Union-find over the graph with an edge wherever dist <= threshold."""
    n = dist.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.nonzero(np.triu(dist <= threshold, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    # relabel to 0..k-1 in order of first appearance
    _, labels = np.unique(roots, return_inverse=True)
    return labels


@dataclass
class DedupResult:
    keep: list[int]  # indices of records kept
    removed_duplicates: list[int]  # extra copies dropped (one copy of each cluster kept)
    removed_conflicts: list[int]  # all members of clusters whose labels disagree
    exact_duplicate_files: int  # byte-identical files beyond the first copy
    clusters_with_duplicates: int
    conflicting_clusters: int


def deduplicate(
    labels: list[str],
    bits: np.ndarray,
    file_md5: list[str],
    sizes: list[int],
    threshold: int = config.DUP_THRESHOLD,
) -> DedupResult:
    """Collapse near-duplicate clusters (pHash distance <= threshold) to one image.

    Clusters whose members carry different labels are dropped entirely, because
    the correct label is unknown. Within a clean cluster the highest-resolution
    image is kept (ties broken by original order).
    """
    dist = hamming_matrix(bits)
    comp = connected_components(dist, threshold)
    keep, dups, conflicts = [], [], []
    n_dup_clusters = n_conflict = 0
    for c in np.unique(comp):
        members = np.nonzero(comp == c)[0].tolist()
        if len(members) == 1:
            keep.append(members[0])
            continue
        if len({labels[m] for m in members}) > 1:
            conflicts.extend(members)
            n_conflict += 1
            continue
        n_dup_clusters += 1
        best = max(members, key=lambda m: (sizes[m], -m))
        keep.append(best)
        dups.extend(m for m in members if m != best)
    exact = len(file_md5) - len(set(file_md5))
    return DedupResult(sorted(keep), sorted(dups), sorted(conflicts), exact, n_dup_clusters, n_conflict)


# ---------------------------------------------------------------- splitting


def group_split(
    labels: list[str],
    groups: np.ndarray,
    test_fraction: float = config.TEST_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
) -> np.ndarray:
    """Return an array of "train" / "val" / "test" such that no group spans two splits.

    Stratified on the (4-class) label so both the binary and 4-class tasks
    see similar class ratios in each split.
    """
    y = np.asarray(labels)
    idx = np.arange(len(y))
    out = np.empty(len(y), dtype=object)

    k_test = max(2, round(1 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    rest, test = next(sgkf.split(idx, y, groups))
    out[test] = "test"

    k_val = max(2, round((1 - test_fraction) / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    tr, va = next(sgkf.split(rest, y[rest], groups[rest]))
    out[rest[tr]] = "train"
    out[rest[va]] = "val"
    return out


# ---------------------------------------------------------------- preprocessing


def load_rgb(path: Path | str) -> np.ndarray:
    """Decode an image file to an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize(rgb: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Bilinear resize (half-pixel centers, no antialiasing) to (size, size, 3) float32.

    This is TensorFlow's default `tf.image.resize` behaviour and matches
    `tf.image.resizeBilinear(x, size, false, true)` in TF.js, which the web demo uses.
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(rgb, dtype=tf.float32)
    return tf.image.resize(x, (size, size), method="bilinear", antialias=False).numpy()


def preprocess_file(path: Path | str, size: int = config.IMG_SIZE) -> np.ndarray:
    """File -> float32 (size, size, 3) array in [0, 255]; the model input."""
    return resize(load_rgb(path), size)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- split manifest


MANIFEST_FIELDS = ["path", "source_split", "label4", "tumor", "group", "split"]


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path = config.DATA_DIR / "splits.csv") -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["tumor"] = int(r["tumor"])
        r["group"] = int(r["group"])
    return rows


def load_split_arrays(split: str, rows: list[dict] | None = None):
    """Load the cached uint8 image tensor plus labels for one split."""
    rows = rows if rows is not None else read_manifest()
    cache = np.load(config.CACHE_DIR / f"images_{config.IMG_SIZE}.npz")
    index = {p: i for i, p in enumerate(cache["paths"])}
    sel = [r for r in rows if r["split"] == split]
    x = cache["images"][[index[r["path"]] for r in sel]]
    y2 = np.array([r["tumor"] for r in sel], dtype=np.int32)
    y4 = np.array([config.CLASSES_4.index(r["label4"]) for r in sel], dtype=np.int32)
    return x, y2, y4, sel
