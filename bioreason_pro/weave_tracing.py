"""Searchable Weave rollout tracing wired to the TRL reward hook (AUTHORED). PROTECTED.

TRL's GRPOTrainer generates completions internally (colocate vLLM) and then calls
reward_funcs(prompts, completions, **cols). That reward callback is where we see every rollout,
so it is the trace point: `make_traced_reward_fn` returns a TRL-shaped reward_fn that scores each
completion inside a `weave.attributes` context and records it as a @weave.op call.

Searchability:
- @weave.op inputs (indexed, queryable as inputs.<name>): sample_id, group_id, rollout_index,
  rl_global_step. The completion/prompt/true_go are logged too (small, per-sample).
- weave.attributes (run-scoped, searchable without signature change): stage, wandb_run_id,
  checkpoint_version, rl_algo, importance_sampling_level.
- The large obo_ancestors / ia maps are captured by CLOSURE, never passed as op inputs (they must
  not be serialized into every trace).
- op.call() returns a (result, Call) TUPLE — unpack it; call.output == result.
"""

from __future__ import annotations

from dataclasses import dataclass

import weave

from bioreason_pro import rewards as R


@dataclass
class RolloutCtx:
    stage: str = "rl"
    wandb_run_id: str = ""
    ckpt_version: str = ""
    rl_algo: str = "grpo"                     # "grpo" (baseline) | "gspo" (experiment)
    importance_sampling_level: str = "token"  # "token" for grpo, "sequence" for gspo


def make_traced_reward_fn(obo_ancestors: dict[str, set[str]], ia: dict[str, float] | None,
                          ctx: RolloutCtx, weights: R.RewardWeights | None = None,
                          final_answer_only: bool = False,
                          go_aspects: dict[str, str] | None = None,
                          go_names: dict[str, str] | None = None):
    """Return a TRL reward_fn(prompts, completions, **cols) -> list[float] that traces each rollout.

    Rewards are identical to bioreason_pro.rewards.make_reward_fn; this adds the searchable Weave
    trace tree on top. obo_ancestors/ia/weights are closed over (not logged).

    The trace carries the reasoning text and its scores as first-class indexed inputs, because the
    point of ADR-014 is that a reviewer can go and read the reasoning behind any prediction. A score
    nobody can trace back to an argument is not what this project is trying to produce.
    """
    w = weights or R.RewardWeights()
    # None keeps the shipped union reward; a mapping selects the aspect-mean fmax term.
    aspects = go_aspects

    # A PURE logging op: it just records the (already-computed) rollout so Weave indexes it. It never
    # computes the reward, so a Weave hiccup can never corrupt/crash the reward the trainer needs.
    @weave.op()  # inputs are indexed / queryable as inputs.<name>
    def rollout(sample_id, group_id, rollout_index, rl_global_step, prompt, completion,
                reward, r_fmax, r_format, r_conciseness, pred_go,
                reasoning, r_substance, r_faithfulness, r_truncated) -> dict:
        return {"reward": reward, "r_fmax": r_fmax, "r_format": r_format,
                "r_conciseness": r_conciseness, "pred_go": pred_go,
                "r_substance": r_substance, "r_faithfulness": r_faithfulness,
                "r_truncated": r_truncated}

    def reward_fn(prompts, completions, **cols):
        gts = R._ground_truth_go(cols)
        ids = cols.get("protein_id") or cols.get("sample_id") or list(range(len(completions)))
        step = int(cols["rl_global_step"][0]) if cols.get("rl_global_step") else 0
        rewards = []
        for i, comp in enumerate(completions):
            # Reward computed DIRECTLY — authoritative, never depends on Weave.
            true_go = set(gts[i]) if i < len(gts) else set()
            components = R.reward_components(
                comp, true_go, obo_ancestors, ia, w, final_answer_only, go_aspects=aspects,
                go_names=go_names,
            )
            rf, fmt, conc, reward = (
                components.fmax,
                components.format,
                components.conciseness,
                components.total,
            )
            rewards.append(reward)
            # Best-effort searchable trace (observability only; never breaks the reward).
            try:
                sid = str(ids[i]) if i < len(ids) else str(i)
                with weave.attributes({
                    "stage": ctx.stage, "wandb_run_id": ctx.wandb_run_id,
                    "checkpoint_version": ctx.ckpt_version, "rl_global_step": step,
                    "rl_algo": ctx.rl_algo, "importance_sampling_level": ctx.importance_sampling_level,
                }):
                    _out, call = rollout.call(
                        sample_id=sid, group_id=f"{ctx.wandb_run_id}-s{step}-{sid}", rollout_index=i,
                        rl_global_step=step, prompt=prompts[i] if i < len(prompts) else "",
                        completion=comp, reward=reward, r_fmax=rf, r_format=fmt, r_conciseness=conc,
                        pred_go=sorted(R.extract_go_terms(comp, final_answer_only)),
                        reasoning=R.reasoning_text(comp),
                        r_substance=components.substance,
                        r_faithfulness=components.faithfulness,
                        r_truncated=components.truncation)
                    call.set_display_name(
                        f"rl s{step} {sid}#{i} r={reward:.3f} f={components.faithfulness:.2f}")
            except Exception:
                pass  # tracing is best-effort; never break the reward the trainer needs
        return rewards

    return reward_fn
