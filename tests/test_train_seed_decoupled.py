"""Varying the training seed must not repartition the data.

The partition seed fixes train/validation membership. A separate training seed varies LoRA
initialization, dropout, and rollout sampling so replicate runs measure training variance
on the same split.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train.py").read_text(encoding="utf-8")

# Call sites that decide WHICH DATA a run sees. These must keep using the partitioning seed.
DATA_PARTITIONING = (
    "data.in_split(r[\"protein_id\"], split, args.seed)",
    "data_contract_json(args.stage, args.smoke, args.seed)",
)


def test_default_reproduces_the_shipped_behaviour():
    import sys

    sys.path.insert(0, str(ROOT))
    from train import RunArgs

    assert RunArgs().train_seed == -1
    assert RunArgs(seed=7).resolved_train_seed() == 7, "default must follow `seed`"
    assert RunArgs(seed=7, train_seed=3).resolved_train_seed() == 3


def test_data_partitioning_still_uses_the_frozen_seed():
    for site in DATA_PARTITIONING:
        assert site in TRAIN, f"data-partitioning site changed: {site}"
        assert site.replace("args.seed", "args.resolved_train_seed()") not in TRAIN


def test_eval_subset_selection_uses_the_frozen_seed():
    """The val-256 subset must be the same proteins across seed replicates."""
    block = TRAIN[TRAIN.index("stream_eval_records(") :][:300]
    assert "seed=args.seed," in block
    assert "resolved_train_seed" not in block


def test_stochastic_training_follows_the_train_seed():
    """Both RNG entry points must vary: global RNG (LoRA init, sampling) and the TRL trainer seed."""
    assert "_seed_everything(args.resolved_train_seed())" in TRAIN
    grpo_cfg = TRAIN[TRAIN.index("def build_grpo_config") :]
    grpo_cfg = grpo_cfg[: grpo_cfg.index("\ndef ")]
    assert "seed=args.resolved_train_seed()," in grpo_cfg
    assert "seed=args.seed," not in grpo_cfg


def test_no_remaining_bare_seed_use_in_stochastic_paths():
    """Catch a future edit that reintroduces args.seed where training randomness is set."""
    for match in re.finditer(r"_seed_everything\((.+)\)\s*$", TRAIN, re.MULTILINE):
        arg = match.group(1)
        if arg.startswith("seed"):  # the function's own definition: _seed_everything(seed: int)
            continue
        assert arg == "args.resolved_train_seed()", f"_seed_everything({arg})"
