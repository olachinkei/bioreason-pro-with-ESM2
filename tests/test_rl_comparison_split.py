"""The RL comparison must not spend the sealed holdout on every GRPO run.

`slurm/rl_grpo.sbatch` runs `compare_rl_checkpoints.py` after publishing, on every
invocation. That script hardcoded `split="test"`, so each RL experiment read 64 proteins of the
sealed public holdout — jobs 637, 640 and 641 each did. Test data must never be used for
training, checkpoint selection, or hyperparameter selection, so an iterative search cannot have the
sealed split wired into its inner loop. Reading it is now an explicit opt-in.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_rl_checkpoints.py"
SBATCH = ROOT / "slurm" / "rl_grpo.sbatch"


def test_comparison_split_defaults_to_val():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"--split", choices=("val", "test"), default="val"' in text
    # The split must come from the argument, not be pinned in the shared kwargs.
    assert '"split": args.split' in text
    assert '"split": "test"' not in text


def test_reading_the_sealed_split_is_announced():
    """A one-off publication read is legitimate; a silent one is not."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if args.split == "test":' in text
    assert "SEALED" in text


def test_the_launcher_passes_val_unless_overridden():
    text = SBATCH.read_text(encoding="utf-8")
    assert 'RL_COMPARISON_SPLIT="${RL_COMPARISON_SPLIT:-val}"' in text
    assert '--split "$RL_COMPARISON_SPLIT"' in text
