"""Units for the traced reward wiring (bioreason_pro/weave_tracing.py).

Runs without weave.init (weave op .call() returns a NoOp call and executes the body), so the
reward numbers must equal the untraced composite reward.
"""

from bioreason_pro import rewards as R
from bioreason_pro import weave_tracing as WT


def _cols():
    return {"protein_id": ["P1", "P2"], "go_mf": ["['GO:0003674']", "['GO:0005515']"]}


def _completions():
    return ["<think>t</think> GO:0003674", "wrong GO:9999999"]


def test_traced_reward_matches_untraced():
    obo, ia = {}, None
    ctx = WT.RolloutCtx(stage="rl", wandb_run_id="run123", rl_algo="grpo")
    traced = WT.make_traced_reward_fn(obo, ia, ctx)
    base = R.make_reward_fn(obo, ia)
    prompts, comps, cols = ["p", "p"], _completions(), _cols()
    assert traced(prompts, comps, **cols) == base(prompts, comps, **cols)


def test_traced_reward_runs_and_ranks_without_weave_init():
    ctx = WT.RolloutCtx(wandb_run_id="r", rl_algo="gspo", importance_sampling_level="sequence")
    fn = WT.make_traced_reward_fn({}, None, ctx)
    out = fn(["p", "p"], _completions(), **_cols())
    assert len(out) == 2 and out[0] > out[1]        # correct+formatted beats wrong+unformatted


def test_rollout_ctx_defaults():
    c = WT.RolloutCtx()
    assert c.stage == "rl" and c.rl_algo == "grpo" and c.importance_sampling_level == "token"
