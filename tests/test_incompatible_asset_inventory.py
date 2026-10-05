"""Offline tests for incompatible-asset inventory matching and classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "inventory_incompatible_assets.py"
    spec = importlib.util.spec_from_file_location("inventory_incompatible_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory():
    return _load_module()


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
