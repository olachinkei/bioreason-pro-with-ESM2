"""Every knob a submitter can set must actually reach the training command.

Job 655 was submitted with SFT_ESM_LAYER=24, but sft.sbatch had no such variable and
train.py's esm_layer only comes from --esm_layer. The job ran as an exact duplicate of an earlier one
and had to be cancelled — an unrecognised environment variable is silently ignored, so the failure
looks like a completed experiment rather than an error.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exported_knobs(text: str) -> set[str]:
    return set(re.findall(r'^export ([A-Z][A-Z0-9_]*)="\$\{\1:-', text, re.M))


def _env_reads_in_python() -> str:
    """Knobs may be consumed by reading os.environ instead of being forwarded as a flag."""
    parts = []
    for path in list(ROOT.glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "slurm").glob("*.py")
    ) + sorted((ROOT / "bioreason_pro").glob("*.py")) + sorted((ROOT / "eval_targets").glob("*.py")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_every_exported_phase_knob_is_actually_consumed():
    """An exported default nothing consumes is a knob that silently does nothing.

    Job 655 was submitted with SFT_ESM_LAYER before the script had it: unrecognised environment
    variables are ignored, so the run looked like a completed experiment instead of an error.
    A knob counts as consumed if the script forwards it, or if Python reads it from os.environ.
    """
    python_src = _env_reads_in_python()
    offenders = []
    for path in sorted((ROOT / "slurm").glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        for knob in _exported_knobs(text):
            forwarded = len(re.findall(rf'\${{?{knob}\b', text)) >= 2
            read_by_python = knob in python_src
            if not (forwarded or read_by_python):
                offenders.append(f"{path.name}:{knob}")
    assert not offenders, f"exported but consumed by nothing: {offenders}"


def test_sft_forwards_the_esm_layer_knob():
    text = (ROOT / "slurm" / "sft.sbatch").read_text(encoding="utf-8")
    assert 'SFT_ESM_LAYER="${SFT_ESM_LAYER:-33}"' in text
    assert '--esm_layer "$SFT_ESM_LAYER"' in text
    # Fifth knob gap found by grepping the flag's destination: SFT replication was unreachable.
    assert 'SFT_TRAIN_SEED="${SFT_TRAIN_SEED:--1}"' in text
    assert '--train_seed "$SFT_TRAIN_SEED"' in text


def test_the_smoke_can_set_every_knob_the_full_sft_run_varies():
    """A knob the smoke cannot set cannot be verified before the expensive run.

    The project requires the smallest meaningful smoke before multi-GPU work, and method rule 6
    requires proving a knob took effect. Those two only compose if the smoke launcher accepts the
    same knobs. `SFT_TRAIN_SEED` reached the full launcher (PR #147) but not the smoke, so the
    only way to exercise it was a ~3.5-hour eight-GPU run.
    """
    smoke = (ROOT / "slurm" / "sft_smoke.sbatch").read_text(encoding="utf-8")
    assert 'SFT_TRAIN_SEED="${SFT_TRAIN_SEED:--1}"' in smoke
    assert '--train_seed "$SFT_TRAIN_SEED"' in smoke


def test_rl_forwards_its_documented_knobs():
    text = (ROOT / "slurm" / "rl_grpo.sbatch").read_text(encoding="utf-8")
    for knob, flag in (
        ("RL_MAX_STEPS", "--max_steps"),
        ("RL_MAX_COMPLETION_LENGTH", "--max_completion_length"),
        ("RL_BETA", "--beta"),
        ("RL_COMPARISON_SPLIT", "--split"),
        ("RL_LORA_R", "--lora_r"),
        ("RL_LORA_ALPHA", "--lora_alpha"),
        ("RL_ALGO", "--rl_algo"),
        ("RL_TRAIN_SEED", "--train_seed"),
    ):
        assert knob in text, knob
        assert flag in text, flag


def test_sft_and_rl_can_both_set_the_architecture_knobs():
    """RL continues an SFT artifact, and train.py rejects an architecture mismatch.

    Job 677 died with "SFT artifact ESM layer 24 does not match requested 33" because
    sft.sbatch could set esm_layer but rl_grpo.sbatch could not — so any non-default
    layer made the RL leg unreachable. Any architecture knob settable for SFT must be settable for
    RL too, or the two launchers cannot agree.
    """
    sft = (ROOT / "slurm" / "sft.sbatch").read_text(encoding="utf-8")
    rl = (ROOT / "slurm" / "rl_grpo.sbatch").read_text(encoding="utf-8")
    for flag in ("--esm_layer",):
        assert flag in sft, f"sft.sbatch must be able to set {flag}"
        assert flag in rl, f"rl_grpo.sbatch must be able to set {flag} to match the SFT artifact"
