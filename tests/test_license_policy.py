"""Offline contract tests for the repository's model-license boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioreason_pro.license_policy import (
    LicensePolicyError,
    approved_model_names,
    load_approved_assets,
    require_approved_dataset,
    require_approved_local_eval_target,
    validate_reference_assets,
    validate_checkpoint_layout,
)
from bioreason_pro.model import ModelConfig, build_model, build_text_model
from train import RunArgs


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_pinned_and_commercially_permissive():
    manifest = load_approved_assets()
    allowed_licenses = set(manifest["policy"]["allowed_spdx_licenses"])
    assert manifest["policy"]["unknown_assets"] == "deny"
    assert allowed_licenses == {"Apache-2.0", "MIT", "CC-BY-4.0"}

    for kind, models in manifest["models"].items():
        assert models, kind
        for model_name, spec in models.items():
            assert spec["license"] in allowed_licenses, model_name
            assert len(spec["revision"]) == 40, model_name
            assert spec["source"].startswith("https://huggingface.co/")
            assert spec["provenance"]
            assert spec["local_cache_policy"]
            assert spec["approved_uses"]
    for repo_id, spec in manifest["datasets"].items():
        assert spec["license"] in allowed_licenses, repo_id
        assert len(spec["revision"]) == 40, repo_id
        assert spec["source"].startswith("https://huggingface.co/datasets/")
        assert spec["provenance"]
        assert spec["local_cache_policy"]
    for name, spec in manifest["reference_data"].items():
        assert spec["license"] in allowed_licenses, name
        assert spec["source"], name
        assert spec["approved_uses"], name


def test_defaults_and_configs_use_only_allowlisted_models():
    default_text = ModelConfig.text_model_name
    assert default_text in approved_model_names("text")
    assert ModelConfig().esm_model_name in approved_model_names("protein")
    assert RunArgs().esm_model_name in approved_model_names("protein")


def test_dataset_uses_are_approved_only_for_the_reviewed_contract():
    for repo, use in (
        ("wanglab/bioreason-pro-test-data", "holdout-evaluation"),
        ("wanglab/bioreason-pro-sft-reasoning-data", "sft-training"),
        ("wanglab/bioreason-pro-sft-reasoning-data", "baseline-evaluation"),
        ("wanglab/bioreason-pro-rl-reasoning-data", "rl-training"),
    ):
        assert len(require_approved_dataset(repo, use)) == 40
    with pytest.raises(LicensePolicyError, match="not approved"):
        require_approved_dataset("wanglab/bioreason-pro-sft-reasoning-data", "evaluation")
    RunArgs(stage="rl", smoke=False, use_multimodal=False).validate()
    RunArgs(
        stage="rl",
        smoke=True,
        sft_artifact=(
            "wandb-healthcare/bioreasonpro-senpai/"
            "bioreasonpro-sft-checkpoint:v0"
        ),
    ).validate()


def test_local_eval_target_approved_for_its_recorded_use():
    spec = require_approved_local_eval_target("cafa_no_knowledge", "dev-evaluation")
    assert spec["approved"] is True
    with pytest.raises(LicensePolicyError, match="not approved"):
        require_approved_local_eval_target("cafa_no_knowledge", "holdout-evaluation")


def test_local_eval_target_unknown_name_denies():
    with pytest.raises(LicensePolicyError, match="not recorded"):
        require_approved_local_eval_target("does-not-exist", "dev-evaluation")


def test_noncommercial_model_package_is_not_a_direct_dependency():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"esm>=' not in pyproject
    assert "override-dependencies" not in pyproject


@pytest.mark.parametrize("model_name", ["esm3_sm_open_v1", "esmc_600m", "unknown/model"])
def test_text_model_rejected_before_transformers_import(model_name):
    with pytest.raises(LicensePolicyError, match="not approved"):
        build_text_model(ModelConfig(), model_name)


def test_unapproved_go_embeddings_fail_before_model_loading():
    cfg = ModelConfig(go_embeddings_path="/tmp/unreviewed-go-embeddings")
    with pytest.raises(LicensePolicyError, match="not yet approved"):
        build_model(cfg)


# --- require_approved_model (plan.md Phase 5: gogpt-baseline's own fail-closed gate) -------------

def test_require_approved_model_accepts_a_listed_name():
    from bioreason_pro.license_policy import require_approved_model

    assert require_approved_model("facebook/esm2_t33_650M_UR50D", "protein") == (
        "facebook/esm2_t33_650M_UR50D"
    )


def test_require_approved_model_rejects_an_unlisted_name():
    from bioreason_pro.license_policy import require_approved_model

    with pytest.raises(LicensePolicyError, match="not approved"):
        require_approved_model("some/unreviewed-model", "protein")


def test_require_approved_model_rejects_an_unknown_kind():
    from bioreason_pro.license_policy import require_approved_model

    with pytest.raises(LicensePolicyError, match="Unknown model kind"):
        require_approved_model("wanglab/gogpt", "not-a-real-kind")


def test_gogpt_and_its_encoder_are_both_approved_for_the_baseline():
    """scripts/run_gogpt_baseline.py checks both wanglab/gogpt itself and whatever encoder its
    downloaded config.yaml declares (facebook/esm2_t36_3B_UR50D, confirmed against the live
    checkpoint) -- both need a reviewed entry, not just the top-level model."""
    from bioreason_pro.license_policy import require_approved_model

    assert require_approved_model("wanglab/gogpt", "go_decoder") == "wanglab/gogpt"
    assert require_approved_model("facebook/esm2_t36_3B_UR50D", "protein") == (
        "facebook/esm2_t36_3B_UR50D"
    )


def test_require_approved_model_is_not_a_revision():
    """Regression (job 1295 smoke test): scripts/run_gogpt_baseline.py once passed
    require_approved_model's return value -- the normalized model NAME -- straight to
    hf_hub_download's `revision=` argument. It 404'd immediately rather than fetching a wrong
    revision silently, but the fix is to call approved_model_revision instead. Pin the shape
    difference so the same mix-up trips a fast, clear assertion next time, on any model."""
    import re

    from bioreason_pro.license_policy import approved_model_revision, require_approved_model

    name = require_approved_model("wanglab/gogpt", "go_decoder")
    revision = approved_model_revision(name, "go_decoder")
    assert re.fullmatch(r"[0-9a-f]{40}", revision), f"not a commit sha: {revision!r}"
    assert revision != name


def _write_checkpoint(
    root: Path,
    *,
    text_model: str = "Qwen/Qwen3-4B-Thinking-2507",
    protein_model: str = "facebook/esm2_t33_650M_UR50D",
    stage: str = "sft",
) -> None:
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": text_model}), encoding="utf-8"
    )
    (root / "run_args.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "model_name": text_model,
                "esm_model_name": protein_model,
                "use_multimodal": True,
                "asset_manifest_schema_version": 2,
                "text_model_revision": {
                    "Qwen/Qwen3-4B-Thinking-2507": "768f209d9ea81521153ed38c47d515654e938aea",
                    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
                }.get(text_model, "unknown"),
                "esm_model_revision": {
                    "facebook/esm2_t33_650M_UR50D": "08e4846e537177426273712802403f7ba8261b6c"
                }.get(protein_model, "unknown"),
                "sft_artifact": (
                    "wandb-healthcare/bioreasonpro-senpai/"
                    "bioreasonpro-sft-checkpoint:v1"
                    if stage == "rl"
                    else ""
                ),
                "resolved_sft_artifact": (
                    "wandb-healthcare/bioreasonpro-senpai/"
                    "bioreasonpro-sft-checkpoint:v1"
                    if stage == "rl"
                    else ""
                ),
                "sft_adapter": "sft_adapter" if stage == "rl" else "",
            }
        ),
        encoding="utf-8",
    )
    if stage == "rl":
        _write_checkpoint(
            root / "sft_adapter",
            text_model=text_model,
            protein_model=protein_model,
            stage="sft",
        )
    (root / "projections.pt").write_bytes(b"presence is checked before tensor loading")
    (root / "data_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "contract_id": "sequence-go-labels-v1",
                "approved_prompt_fields": ["sequence"],
                "approved_label_fields": ["go_mf", "go_bp", "go_cc"],
                "preprocessing": {
                    "max_protein_residues": 1024,
                    "protein_special_tokens": ["BOS", "EOS"],
                    "max_protein_tokens": 1026,
                },
            }
        ),
        encoding="utf-8",
    )


def test_approved_adapter_checkpoint_metadata_passes(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    metadata = validate_checkpoint_layout(checkpoint)
    assert metadata["esm_model_name"] == "facebook/esm2_t33_650M_UR50D"


def test_reloadable_rl_checkpoint_requires_bundled_sft_chain(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, stage="rl")
    assert validate_checkpoint_layout(checkpoint)["stage"] == "rl"
    (checkpoint / "sft_adapter" / "adapter_config.json").unlink()
    with pytest.raises(LicensePolicyError, match="missing required"):
        validate_checkpoint_layout(checkpoint)


@pytest.mark.parametrize("protein_model", ["esm3_sm_open_v1", "esmc_600m", "unknown/model"])
def test_checkpoint_rejects_unapproved_protein_metadata(tmp_path, protein_model):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, protein_model=protein_model)
    with pytest.raises(LicensePolicyError, match="not approved"):
        validate_checkpoint_layout(checkpoint)


def test_checkpoint_rejects_known_full_model_layout(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "protein_model").mkdir()
    with pytest.raises(LicensePolicyError, match="bundled protein_model"):
        validate_checkpoint_layout(checkpoint)


def test_checkpoint_rejects_adapter_and_run_metadata_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-0.6B"}), encoding="utf-8"
    )
    with pytest.raises(LicensePolicyError, match="does not match run_args"):
        validate_checkpoint_layout(checkpoint)


def test_checkpoint_rejects_requested_architecture_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, text_model="Qwen/Qwen3-0.6B")
    with pytest.raises(LicensePolicyError, match="does not match requested"):
        validate_checkpoint_layout(
            checkpoint, expected_text_model="Qwen/Qwen3-4B-Thinking-2507"
        )


@pytest.mark.parametrize(
    "missing", ["adapter_config.json", "run_args.json", "data_manifest.json", "projections.pt"]
)
def test_checkpoint_rejects_incomplete_layout(tmp_path, missing):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / missing).unlink()
    with pytest.raises(LicensePolicyError, match="missing required"):
        validate_checkpoint_layout(checkpoint)


def test_reference_assets_fail_closed_when_missing(tmp_path):
    with pytest.raises(LicensePolicyError, match="fetch_approved_reference_data"):
        validate_reference_assets(tmp_path, use="reward")


def test_checkpoint_rejects_old_2000_residue_preprocessing(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    manifest_path = checkpoint / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing"]["max_protein_residues"] = 2000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LicensePolicyError, match="ESM2 limits"):
        validate_checkpoint_layout(checkpoint)


def test_cafa5_remains_unavailable_in_manifest():
    manifest = load_approved_assets()
    assert "wanglab/cafa5" not in manifest["datasets"]


def test_real_multimodal_rl_requires_versioned_sft_artifact():
    with pytest.raises(ValueError, match="--sft_artifact"):
        RunArgs(stage="rl", smoke=False, use_multimodal=True).validate()
