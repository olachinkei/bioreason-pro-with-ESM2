"""Unit tests for scripts/run_gogpt_baseline.py's pure parts (no network, no model load)."""

from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from run_gogpt_baseline import _row_gt, union_predicted_terms  # noqa: E402


def test_union_predicted_terms_flattens_all_three_aspects():
    predictions = {"MF": ["GO:0003674", "GO:0005524"], "BP": ["GO:0008152"], "CC": []}
    assert union_predicted_terms(predictions) == {"GO:0003674", "GO:0005524", "GO:0008152"}


def test_union_predicted_terms_dedupes_across_aspects():
    predictions = {"MF": ["GO:0003674"], "BP": ["GO:0003674"], "CC": ["GO:0003674"]}
    assert union_predicted_terms(predictions) == {"GO:0003674"}


def test_union_predicted_terms_handles_all_empty():
    assert union_predicted_terms({"MF": [], "BP": [], "CC": []}) == set()


def test_row_gt_unions_the_three_ground_truth_columns():
    row = {"go_mf": ["GO:0000001"], "go_bp": ["GO:0000002"], "go_cc": []}
    assert _row_gt(row) == {"GO:0000001", "GO:0000002"}


def test_row_gt_handles_missing_columns():
    assert _row_gt({}) == set()
