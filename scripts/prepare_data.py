"""Download the dataset, deduplicate with perceptual hashing, split, and cache resized images.

Outputs:
  data/splits.csv                 one row per kept image with its split
  data/cache/images_224.npz       uint8 (N, 224, 224, 3) tensor (not committed)
  reports/dedup.json              duplicate and split statistics
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from PIL import Image

from btc import config
from btc.data import (
    connected_components,
    deduplicate,
    download,
    group_split,
    hamming_matrix,
    list_records,
    phash_bits,
    preprocess_file,
    to_uint8,
    write_manifest,
)


def main() -> None:
    download()
    recs = list_records()
    print(f"{len(recs)} images found")

    bits, md5s, sizes = [], [], []
    for r in recs:
        p = config.RAW_DIR / r.path
        md5s.append(hashlib.md5(p.read_bytes()).hexdigest())
        with Image.open(p) as im:
            sizes.append(im.size[0] * im.size[1])
            bits.append(phash_bits(im))
    bits = np.array(bits)
    labels4 = [r.label4 for r in recs]

    # How much the original Training/Testing folders overlap.
    dist_all = hamming_matrix(bits)
    train_idx = [i for i, r in enumerate(recs) if r.source_split == "Training"]
    test_idx = [i for i, r in enumerate(recs) if r.source_split == "Testing"]
    cross = dist_all[np.ix_(test_idx, train_idx)]
    orig_test_with_dup_in_train = int((cross.min(1) <= config.DUP_THRESHOLD).sum())
    orig_test_with_near_in_train = int((cross.min(1) <= config.GROUP_THRESHOLD).sum())

    res = deduplicate(labels4, bits, md5s, sizes, config.DUP_THRESHOLD)
    kept = [recs[i] for i in res.keep]
    kept_bits = bits[res.keep]
    print(
        f"removed {len(res.removed_duplicates)} duplicate copies "
        f"({res.clusters_with_duplicates} clusters) and {len(res.removed_conflicts)} images "
        f"in {res.conflicting_clusters} label-conflicting clusters; {len(kept)} kept"
    )

    # Near-identical images (adjacent slices of one series, re-crops) are grouped
    # so a group never spans train/val/test.
    groups = connected_components(hamming_matrix(kept_bits), config.GROUP_THRESHOLD)
    splits = group_split([r.label4 for r in kept], groups)

    rows = [
        dict(path=r.path, source_split=r.source_split, label4=r.label4, tumor=r.tumor, group=int(g), split=s)
        for r, g, s in zip(kept, groups, splits)
    ]
    write_manifest(config.DATA_DIR / "splits.csv", rows)

    # Sanity check: no near-duplicate pair crosses a split boundary.
    kd = hamming_matrix(kept_bits)
    s = np.asarray(splits)
    cross_pairs = int(((kd <= config.GROUP_THRESHOLD) & (s[:, None] != s[None, :])).sum() // 2)
    assert cross_pairs == 0, cross_pairs

    def counts(split=None):
        sel = [r for r in rows if split is None or r["split"] == split]
        c4 = Counter(r["label4"] for r in sel)
        return {
            "n": len(sel),
            "healthy": sum(1 for r in sel if r["tumor"] == 0),
            "tumor": sum(1 for r in sel if r["tumor"] == 1),
            "by_class": dict(sorted(c4.items())),
        }

    report = {
        "dataset": config.HF_REPO,
        "revision": config.HF_REVISION,
        "images_downloaded": len(recs),
        "images_by_class_downloaded": dict(sorted(Counter(labels4).items())),
        "hash": "64-bit pHash (imagehash.phash on grayscale), Hamming distance",
        "dup_threshold": config.DUP_THRESHOLD,
        "group_threshold": config.GROUP_THRESHOLD,
        "exact_duplicate_files": res.exact_duplicate_files,
        "near_duplicate_clusters": res.clusters_with_duplicates,
        "duplicate_copies_removed": len(res.removed_duplicates),
        "label_conflict_clusters": res.conflicting_clusters,
        "label_conflict_images_removed": len(res.removed_conflicts),
        "images_kept": len(kept),
        "original_testing_images": len(test_idx),
        "original_testing_with_duplicate_in_training": orig_test_with_dup_in_train,
        "original_testing_with_near_duplicate_in_training": orig_test_with_near_in_train,
        "groups": int(groups.max() + 1),
        "largest_group": int(np.bincount(groups).max()),
        "near_duplicate_pairs_across_new_splits": cross_pairs,
        "splits": {k: counts(k) for k in ("train", "val", "test")},
        "total": counts(),
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "dedup.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    # Cache resized images for training.
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    imgs = np.stack([to_uint8(preprocess_file(config.RAW_DIR / r["path"])) for r in rows])
    np.savez(config.CACHE_DIR / f"images_{config.IMG_SIZE}.npz", images=imgs, paths=np.array([r["path"] for r in rows]))
    print("cached", imgs.shape)


if __name__ == "__main__":
    main()
