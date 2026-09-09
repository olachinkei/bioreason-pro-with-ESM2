"""Offline tests for the Phase 0 incompatible-asset inventory and its quarantine claim."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "incompatible_asset_inventory.json"


def _load_module():
    path = ROOT / "scripts" / "inventory_incompatible_assets.py"
    spec = importlib.util.spec_from_file_location("inventory_incompatible_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory():
    return _load_module()


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("esm_model_name: esm3_sm_open_v1", ["esm3"]),
        ("model: esmc_600m", ["esmc"]),
        ("hf://wanglab/bioreason-pro-sft", ["released_paper_checkpoint"]),
        ("go_cached_embedding_path: /shared/go.pt", ["unapproved_go_embeddings"]),
        # The approved assets must never match.
        ("facebook/esm2_t33_650M_UR50D", []),
        ("wanglab/bioreason-pro-sft-reasoning-data", []),
        ("wanglab/bioreason-pro-test-data", []),
        ("Qwen/Qwen3-4B-Thinking-2507", []),
    ],
)
def test_rules_match_incompatible_assets_and_spare_approved_ones(inventory, text, expected):
    assert inventory._matched_rules(text) == expected


def test_gated_dataset_rule_is_filesystem_only(inventory):
    cached = "datasets--wanglab--cafa5"
    assert inventory._matched_rules(cached) == []
    assert inventory._matched_rules(cached, include_filesystem_only=True) == [
        "gated_dataset_cache"
    ]


@pytest.mark.parametrize(
    "path, text, expected",
    [
        ("bioreason_pro/license_policy.py", "esm3_sm_open_v1", "policy_reference"),
        ("tests/test_license_policy.py", "esm3_sm_open_v1", "policy_reference"),
        ("configs/stage3_rl.yaml", "# Not comparable to the ESM3 result.", "commentary"),
        ("configs/stage3_rl.yaml", "esm_model_name: esm3_sm_open_v1", "asset_reference"),
    ],
)
def test_classification_separates_denial_from_a_live_asset_route(inventory, path, text, expected):
    assert inventory._classify(path, text) == expected


def test_report_records_which_scanners_and_roots_actually_ran(report):
    """An empty result must never be readable as `clean` for a root nobody scanned."""
    assert report["schema_version"] == 1
    assert set(report["scanners_run"]) == {"git", "filesystem", "wandb"}
    assert report["filesystem_roots_scanned"]


def test_main_carries_no_live_route_to_an_incompatible_asset(report):
    """The boundary landed on main; this is the regression guard for it."""
    main_refs = {"refs/heads/main", "refs/remotes/origin/main"}
    offenders = [
        item
        for item in report["requiring_quarantine"]
        if item["source"] == "git" and item.get("ref") in main_refs
    ]
    assert offenders == [], offenders


def test_inventoried_wandb_artifacts_are_absent_from_every_launch_path(report):
    """Quarantine claim: nothing on a launch path names an inventoried artifact."""
    names = {
        item["artifact"].split(":")[0]
        for item in report["requiring_quarantine"]
        if item.get("artifact")
    }
    assert names, "the inventory should have recorded incompatible W&B artifacts"

    launch_paths = [
        *(ROOT / "configs").glob("*.yaml"),
        *(ROOT / "slurm").rglob("*.sbatch"),
        *(ROOT / "slurm").rglob("*.tmpl"),
        *(ROOT / "slurm").glob("*.py"),
        *(ROOT / "slurm").glob("*.yaml"),
        ROOT / "train.py",
        ROOT / "eval.py",
        ROOT / "senpai.yaml",
        ROOT / "agent_policy.yaml",
        ROOT / "bioreason_pro" / "approved_assets.json",
    ]
    for path in launch_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for name in names:
            assert name not in text, f"{path} references quarantined artifact {name}"
