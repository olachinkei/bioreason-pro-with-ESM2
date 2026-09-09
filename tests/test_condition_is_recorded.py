"""Every condition the search varies must be recoverable from the W&B config.

plan.md states the principle: "a condition search whose runs do not record their condition is not
auditable." `_init_tracking` recorded `target_variant` and `prompt_variant` but NOT `reward_variant`,
so the aspect_mean-versus-union arms — the ledger's largest single RL claim — were distinguishable
only from each job's stdout. Runs 9ygnae6t and 2tlj7noq carry no reward_variant key at all.

The reward variant is recorded for RL runs only: it has no effect on an SFT run, and recording a
default there would assert that a reward was used when none was.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train.py").read_text(encoding="utf-8")


def _init_tracking_body() -> str:
    start = TRAIN.index("def _init_tracking")
    rest = TRAIN[start + 1 :]
    end = rest.index("\ndef ") if "\ndef " in rest else len(rest)
    return TRAIN[start : start + 1 + end]


def test_every_varied_condition_reaches_the_wandb_config():
    body = _init_tracking_body()
    for key in ("target_variant", "prompt_variant", "reward_variant"):
        assert f'"{key}"' in body, f"{key} is not recorded in the W&B config"


def test_the_variants_are_resolved_through_the_fail_closed_readers():
    """A literal env lookup would record an unvalidated string; use the same readers training uses."""
    body = _init_tracking_body()
    for reader in ("active_target_variant()", "active_prompt_variant()", "active_reward_variant()"):
        assert reader in body, f"{reader} not used"


def test_reward_variant_is_recorded_for_rl_only():
    body = _init_tracking_body()
    match = re.search(r'"reward_variant":[^\n]*', body)
    assert match, "reward_variant line not found"
    line = match.group(0)
    assert 'args.stage == "rl"' in line, (
        "reward_variant must be gated on the RL stage; recording it on an SFT run asserts a reward "
        f"was used when none was. Got: {line}"
    )
