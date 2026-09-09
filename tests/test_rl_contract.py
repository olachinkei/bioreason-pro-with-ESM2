"""CPU-only checks for the self-contained SFT→RL checkpoint contract."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import train
from scripts.compare_rl_checkpoints import _evaluate_with_empty_prediction_fallback


class _Core:
    def __init__(self):
        self.protein_projection = nn.Sequential(nn.Linear(2, 2))
        self.go_projection = None


SFT_REF = "wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:v3"


def test_rl_comparison_records_empty_predictions_instead_of_aborting():
    class Evaluator:
        @staticmethod
        def evaluate(checkpoint, **kwargs):
            raise KeyError(
                "cafa_eval returned no 'f_w'/'f' best-scores frame — pass the IA file (ia)."
            )

    result = _evaluate_with_empty_prediction_fallback(
        Evaluator,
        "checkpoint",
        split="test",
    )

    assert result["weighted_fmax"] == 0.0
    assert result["weighted_fmax_mf"] == 0.0
    assert result["weighted_fmax_bp"] == 0.0
    assert result["weighted_fmax_cc"] == 0.0
    assert result["scoring_status"] == "empty_predictions"


def test_rl_checkpoint_bundles_parent_and_records_artifact_lineage(tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    core = _Core()
    out = tmp_path / "rl"
    out.mkdir()
    args = train.RunArgs(
        stage="rl",
        smoke=True,
        model_name="Qwen/Qwen3-0.6B",
        sft_artifact=SFT_REF,
        resolved_sft_artifact=SFT_REF,
        sft_adapter=str(sft),
    )

    train._save_mm_extras(core, args, str(out))

    saved = json.loads((out / "run_args.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "rl"
    assert saved["sft_artifact"] == SFT_REF
    assert saved["resolved_sft_artifact"] == SFT_REF
    assert saved["sft_adapter"] == "sft_adapter"
    assert (out / "sft_adapter" / "adapter_config.json").is_file()
    assert (out / "projections.pt").is_file()
    assert (out / "data_manifest.json").is_file()


def test_projection_state_is_restored_before_rl(tmp_path):
    source = _Core()
    with torch.no_grad():
        source.protein_projection[0].weight.fill_(3.0)
        source.protein_projection[0].bias.fill_(4.0)
    torch.save(
        {"protein_projection": source.protein_projection.state_dict()},
        tmp_path / "projections.pt",
    )
    target = _Core()

    train._load_projection_state(target, str(tmp_path))

    assert torch.equal(
        target.protein_projection[0].weight,
        source.protein_projection[0].weight,
    )
    assert torch.equal(
        target.protein_projection[0].bias,
        source.protein_projection[0].bias,
    )


def test_rl_rejects_an_sft_artifact_from_a_different_text_backbone(monkeypatch):
    monkeypatch.setattr(
        "bioreason_pro.license_policy.validate_checkpoint_layout",
        lambda _path: {
            "model_name": "Qwen/Qwen3-4B-Thinking-2507",
            "esm_model_name": "facebook/esm2_t33_650M_UR50D",
            "esm_layer": 33,
        },
    )

    with pytest.raises(ValueError, match="does not match requested"):
        train._validate_sft_input_architecture(
            train.RunArgs(model_name="Qwen/Qwen3-0.6B"),
            "unused-checkpoint",
        )


def test_use_model_artifact_declares_input_before_download(monkeypatch, tmp_path):
    import bioreason_pro.license_policy as policy

    class Artifact:
        qualified_name = SFT_REF

        def download(self, root):
            return root

    class Run:
        entity = "wandb-healthcare"
        project = "bioreasonpro-senpai"

        def __init__(self):
            self.calls = []

        def use_artifact(self, ref, type):
            self.calls.append((ref, type))
            return Artifact()

    monkeypatch.setattr(policy, "validate_checkpoint_layout", lambda path: {"stage": "sft"})
    run = Run()
    local, resolved = train.use_model_artifact(run, SFT_REF, str(tmp_path / "download"))

    assert run.calls == [(SFT_REF, "model")]
    assert resolved == SFT_REF
    assert local == str((tmp_path / "download").resolve())


def test_artifact_alias_is_rejected():
    import pytest

    from bioreason_pro.license_policy import LicensePolicyError

    with pytest.raises(LicensePolicyError, match="immutable W&B reference"):
        train.RunArgs(
            stage="rl",
            smoke=True,
            sft_artifact=(
                "wandb-healthcare/bioreasonpro-senpai/"
                "bioreasonpro-sft-checkpoint:latest"
            ),
        ).validate()


def test_checkpoint_artifact_waits_and_writes_immutable_receipt(monkeypatch, tmp_path):
    import wandb

    class DraftArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.directory = None

        def add_dir(self, directory):
            self.directory = directory

    class LoggedArtifact:
        qualified_name = (
            "wandb-healthcare/bioreasonpro-senpai/"
            "bioreasonpro-sft-checkpoint:v7"
        )

        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    class Run:
        id = "run123"
        entity = "wandb-healthcare"
        project = "bioreasonpro-senpai"

        def __init__(self):
            self.summary = {}
            self.logged = LoggedArtifact()
            self.aliases = None

        def log_artifact(self, artifact, aliases):
            self.artifact = artifact
            self.aliases = aliases
            return self.logged

    monkeypatch.setattr(wandb, "Artifact", DraftArtifact)
    (tmp_path / "model.bin").write_bytes(b"weights")
    run = Run()

    ref = train._log_checkpoint_artifact(
        str(tmp_path), train.RunArgs(stage="sft", smoke=True), run
    )

    assert run.logged.waited is True
    assert run.aliases == ["latest", "sft"]
    assert ref.endswith("bioreasonpro-sft-checkpoint:v7")
    receipt = json.loads((tmp_path / "wandb_artifact.json").read_text(encoding="utf-8"))
    assert receipt["artifact_ref"] == ref
    assert run.summary["sft_artifact_ref"] == ref
