"""Offline checks for the reviewed sequence/GO-only data boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import data

from bioreason_pro.data_contract import (
    adapt_synthetic_fixture_row,
    assert_train_val_holdout_disjoint,
    build_run_data_manifest,
    public_holdout_ids,
)
from bioreason_pro.license_policy import load_approved_assets


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "approved_training_rows.jsonl"


def _raw_rows():
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_adapter_uses_only_sequence_and_go_labels():
    """The DEFAULT contract: sequence in, GO labels out, nothing else.

    ADR-017 admits `reasoning` and the InterPro/STRING/location context under
    SENPAI_TARGET_VARIANT=leaf_only_reasoned, and only there. This test pins the default, which is
    what every SFT recorded in plan.md trained on — including the three context columns, so a future
    change cannot leak them into the shipped prompt without failing here.
    """
    raw = _raw_rows()[0]
    approved = adapt_synthetic_fixture_row(raw)
    serialized = json.dumps(approved, sort_keys=True)
    assert raw["sequence"] in approved["prompt"]["user"]
    assert approved["prompt"]["assistant_reasoning"] == ""
    assert approved["prompt"]["assistant_answer"] == (
        "MF: GO:0005524\nBP: GO:0008152"
    )
    assert "MUST_NOT_ENTER_PROMPT" not in serialized
    assert not ({"reasoning", "final_answer", "protein_function",
                 "interpro_formatted", "ppi_formatted", "subcellular_location"} & approved.keys())


def test_checked_in_holdout_ids_match_manifest():
    ids = public_holdout_ids()
    spec = load_approved_assets()["data_contract"]["public_holdout_ids"]
    digest = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert len(ids) == spec["count"]
    assert digest == spec["sha256"]


def test_public_holdout_is_excluded_from_every_internal_split():
    protein_id = next(iter(public_holdout_ids()))
    assert all(not data.in_split(protein_id, split) for split in data.SPLITS)


def test_fixture_operational_splits_are_pairwise_disjoint():
    ids = {row["protein_id"] for row in _raw_rows()}
    holdout = set()
    train = {protein_id for protein_id in ids if data.protein_split(protein_id) == "rl_train"}
    validation = {protein_id for protein_id in ids if data.protein_split(protein_id) == "val"}
    assert_train_val_holdout_disjoint(train, validation, holdout)
    assert train and validation


def test_run_manifest_names_every_source_and_field():
    real = build_run_data_manifest("sft", smoke=False, seed=7)
    assert real["source"]["repo_id"] == "wanglab/bioreason-pro-sft-reasoning-data"
    assert len(real["source"]["revision"]) == 40
    assert real["approved_prompt_fields"] == ["sequence"]
    assert real["approved_label_fields"] == ["go_mf", "go_bp", "go_cc"]
    assert real["preprocessing"]["max_protein_residues"] == 1024
    assert real["preprocessing"]["max_protein_tokens"] == 1026
    assert real["split_policy"]["public_holdout_excluded_first"] is True

    smoke = build_run_data_manifest("rl", smoke=True, seed=0)
    assert smoke["source"]["path"] == "tests/fixtures/approved_training_rows.jsonl"
