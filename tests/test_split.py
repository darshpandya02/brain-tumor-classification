import csv
import json
from collections import Counter, defaultdict

import numpy as np

from btc import config
from btc.data import group_split


def test_group_split_never_splits_a_group():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, 1000)
    labels = [["a", "b", "c", "d"][g % 4] for g in groups]
    split = group_split(labels, groups, 0.15, 0.15, seed=1)
    seen = defaultdict(set)
    for g, s in zip(groups, split):
        seen[g].add(s)
    assert all(len(v) == 1 for v in seen.values())
    frac = Counter(split)
    assert 0.08 < frac["test"] / 1000 < 0.25
    assert 0.08 < frac["val"] / 1000 < 0.25


def test_group_split_is_deterministic():
    groups = np.arange(200) // 2
    labels = ["x" if g % 3 else "y" for g in groups]
    a = group_split(labels, groups, seed=5)
    b = group_split(labels, groups, seed=5)
    assert (a == b).all()


def test_committed_manifest_is_consistent_with_dedup_report():
    """The committed split file has no group spanning splits and matches reports/dedup.json."""
    with open(config.DATA_DIR / "splits.csv") as f:
        rows = list(csv.DictReader(f))
    report = json.loads((config.REPORTS_DIR / "dedup.json").read_text())
    assert len(rows) == report["images_kept"]
    by_group = defaultdict(set)
    for r in rows:
        by_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
    assert len({r["path"] for r in rows}) == len(rows)
    for split in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == split]
        assert len(sel) == report["splits"][split]["n"]
        assert sum(r["tumor"] == "0" for r in sel) == report["splits"][split]["healthy"]
    removed = report["duplicate_copies_removed"] + report["label_conflict_images_removed"]
    assert report["images_downloaded"] - removed == report["images_kept"]
