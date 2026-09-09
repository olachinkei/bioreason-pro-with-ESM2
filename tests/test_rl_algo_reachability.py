"""An RL algorithm the code cannot implement must fail, not run as its baseline.

`--rl_algo gspo` was reachable-looking from every angle: RunArgs accepted it, plan.md listed it as an
optional follow-up, and `RL_ALGO` now forwards it. But the multimodal loop
(`_run_multimodal_grpo`) is strictly on-policy — rollouts are generated fresh each step and consumed
by one update — so the importance-sampling ratio is identically 1 and never appears in the loss
(`-(a/G) * pol.mean()`). GRPO and GSPO differ ONLY in how that ratio is aggregated, so on this path
gspo is a bit-for-bit GRPO duplicate.

That would have been undetectable downstream: `reward_ctx` stamps every Weave rollout trace with
importance_sampling_level="sequence", and the W&B config records rl_algo="gspo", so the duplicate run
would have been filed as "GSPO tested, no improvement" — a fabricated null result. Job 691 was
launched and cancelled 9 steps in for exactly this reason.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train.py").read_text(encoding="utf-8")


def _multimodal_rl_body() -> str:
    """Source of `_run_multimodal_grpo` up to the next top-level def."""
    start = TRAIN.index("def _run_multimodal_grpo(")
    rest = TRAIN[start + 1 :]
    end = rest.index("\ndef ") if "\ndef " in rest else len(rest)
    return TRAIN[start : start + 1 + end]


def test_the_multimodal_path_refuses_an_algo_it_cannot_honour():
    body = _multimodal_rl_body()
    assert 'if args.rl_algo != "grpo":' in body
    assert "raise ValueError" in body
    # The message must say WHY, so the next person does not simply relax the check.
    assert "importance-sampling ratio" in body


def test_the_refusal_is_reached_before_any_training_step():
    """A guard that fires after 50 steps still burns the node it was meant to save."""
    body = _multimodal_rl_body()
    guard = body.index('if args.rl_algo != "grpo":')
    for later in ("opt.zero_grad()", "loss_i.backward()", "opt.step()"):
        assert body.index(later) > guard, f"{later} runs before the rl_algo guard"


def test_the_text_only_path_still_implements_gspo_for_real():
    """The guard must not be mistaken for gspo being unsupported everywhere."""
    assert 'importance_sampling_level="sequence"' in TRAIN
    assert 'importance_sampling_level="token"' in TRAIN


def test_logged_reward_components_score_the_objective_being_optimised():
    """rl/reward_fmax and the rollout table must use the same variant as reward_fn.

    `reward_components` defaults go_aspects=None, which is the union reward. Omitting the argument in
    the telemetry call made every aspect_mean run report a union fmax it was not trained on.
    """
    body = _multimodal_rl_body()
    trained = re.search(r"make_traced_reward_fn\((.*?)\)", body, re.DOTALL).group(1)
    assert "go_aspects=go_aspects" in trained

    logged = re.search(r"components = \[\s*rewards\.reward_components\((.*?)\)", body, re.DOTALL)
    assert logged, "could not find the telemetry reward_components call"
    assert "go_aspects=go_aspects" in logged.group(1), (
        "telemetry scores the union reward while training optimises aspect_mean"
    )
