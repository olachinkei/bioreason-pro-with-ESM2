"""CPU-only contracts for the distributed full-holdout comparison."""

from __future__ import annotations

import json

import pytest

from scripts.compare_full_holdout import (
    _expected_count,
    _job_type_for_target,
    _merge_and_validate_shards,
    _parse_labels,
)


def test_job_type_is_sealed_only_for_the_sealed_target():
    assert _job_type_for_target("bioreason_pro_test") == "full-holdout-evaluation"
    assert _job_type_for_target("cafa_no_knowledge") == "dev-set-evaluation"
    assert _job_type_for_target("cafa5") == "dev-set-evaluation"


def test_parse_labels_default_runs_both_checkpoints():
    assert _parse_labels("sft,sft_to_grpo") == ("sft", "sft_to_grpo")


def test_parse_labels_accepts_a_single_checkpoint_for_split_submission():
    assert _parse_labels("sft") == ("sft",)
    assert _parse_labels("sft_to_grpo") == ("sft_to_grpo",)


def test_parse_labels_tolerates_whitespace():
    assert _parse_labels(" sft , sft_to_grpo ") == ("sft", "sft_to_grpo")


def test_parse_labels_rejects_unknown_or_empty():
    with pytest.raises(ValueError, match="must be a comma-separated subset"):
        _parse_labels("rl")
    with pytest.raises(ValueError, match="must be a comma-separated subset"):
        _parse_labels("")


def _row(protein_id: str, response: str = "GO:0003674") -> dict:
    return {
        "protein_id": protein_id,
        "generated_response": response,
        "gt_terms": ["GO:0003674"],
    }


def _write_shard(root, label, rank, rows):
    (root / f"{label}-rank-{rank:02d}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_merge_requires_exact_disjoint_coverage_and_matching_ground_truth(tmp_path):
    for label in ("sft", "sft_to_grpo"):
        _write_shard(tmp_path, label, 0, [_row("P0"), _row("P2")])
        _write_shard(tmp_path, label, 1, [_row("P1"), _row("P3")])

    merged = _merge_and_validate_shards(tmp_path, ("sft", "sft_to_grpo"), 2, 4)

    assert [row["protein_id"] for row in merged["sft"]] == ["P0", "P1", "P2", "P3"]
    assert merged["sft"][0]["gt_terms"] == {"GO:0003674"}


def test_merge_rejects_duplicate_protein_ids(tmp_path):
    _write_shard(tmp_path, "sft", 0, [_row("P0")])
    _write_shard(tmp_path, "sft", 1, [_row("P0")])
    with pytest.raises(ValueError, match="duplicate protein_id"):
        _merge_and_validate_shards(tmp_path, ("sft",), 2, 2)


def test_merge_rejects_incomplete_full_holdout(tmp_path):
    _write_shard(tmp_path, "sft", 0, [_row("P0")])
    with pytest.raises(ValueError, match="coverage is 1, expected exactly 2"):
        _merge_and_validate_shards(tmp_path, ("sft",), 1, 2)


def test_merge_rejects_model_set_drift(tmp_path):
    _write_shard(tmp_path, "sft", 0, [_row("P0")])
    _write_shard(tmp_path, "sft_to_grpo", 0, [_row("P1")])
    with pytest.raises(ValueError, match="differ from the SFT set"):
        _merge_and_validate_shards(tmp_path, ("sft", "sft_to_grpo"), 1, 1)


def test_expected_count_uses_subset_or_full_contract():
    assert _expected_count(4, 8_630) == 4
    assert _expected_count(None, 8_630) == 8_630
    with pytest.raises(ValueError):
        _expected_count(0, 8_630)
