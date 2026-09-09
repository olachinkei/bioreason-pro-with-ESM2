"""Pre-GPU units for the split/leakage contract (data.py). Guards against train/eval leakage."""

import pytest

import data as D


def test_protein_split_is_deterministic_and_stable():
    assert D.protein_split("P12345", seed=0) == D.protein_split("P12345", seed=0)


def test_splits_are_disjoint_by_construction():
    ids = [f"P{i:05d}" for i in range(3000)]
    buckets = {"rl_train": [], "val": [], "test": []}
    for p in ids:
        buckets[D.protein_split(p, seed=0)].append(p)
    # every protein lands in exactly one bucket, and all three are populated
    assert sum(len(v) for v in buckets.values()) == len(ids)
    D.assert_disjoint_splits(buckets["rl_train"], buckets["val"], buckets["test"])
    assert all(len(v) > 0 for v in buckets.values())


def test_assert_disjoint_splits_flags_leakage():
    with pytest.raises(AssertionError):
        D.assert_disjoint_splits(["P1", "P2"], ["P2"], ["P3"])   # P2 in two splits


def test_deterministic_val_subset_is_reproducible_and_capped():
    val = [f"P{i:05d}" for i in range(1000)]
    a = D.deterministic_val_subset(val, size=256, seed=0)
    b = D.deterministic_val_subset(val, size=256, seed=0)
    assert a == b and len(a) == 256
