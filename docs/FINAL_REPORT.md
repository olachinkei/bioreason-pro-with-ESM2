# BioReason-Pro Senpai: final report

What was built, what it scores, and — the part worth reading — which of its conclusions survived being
checked. Decisions and their reasoning are in [`../plan.md`](../plan.md); this document is the result.

**Companion.** [`ROLLOUT_ANALYSIS.md`](ROLLOUT_ANALYSIS.md) reads 472 stored rollouts to show each
mechanism below working — or, in one case, not working at all.

**Objective.** A trustworthy protein-function-prediction workflow under a hard license boundary, not a
competitive CAFA score. No ESM3 or ESM-C 600M weights, outputs, embeddings, or derivatives enter any
run, and the released BioReason-Pro checkpoint is not used even as a baseline (ADR-001). Encoder is
`facebook/esm2_t33_650M_UR50D` (MIT), backbone `Qwen/Qwen3-4B-Thinking-2507` (Apache-2.0). **No number
here is comparable to the paper's.** That is accepted, not worked around.

---

## Headline

| Measurement | Value | Seeds | Split |
|---|---:|---:|---|
| Recommended recipe, sealed holdout @ 64 | **0.17820 ± 0.00830** | 3 | 8,630 sealed proteins |
| Its SFT parent, sealed holdout @ 64 | 0.17068 ± 0.00066 | 1 ckpt, 3 evals | 8,630 sealed proteins |
| Phase 5 baseline (`full_closure` SFT→GRPO @ 1024) | 0.084657 | 1 | 8,630 sealed proteins |
| Recommended recipe, validation @ 64 | 0.27594 ± 0.0069 | 3 | val-256 |
| As-is shipped checkpoint at its shipped budget | 0.11227 | 1 | val-256 |

**Recipe.** `SENPAI_TARGET_VARIANT=leaf_only` + `SENPAI_REWARD_VARIANT=aspect_mean`, GRPO 50 steps,
`num_generations=8`, `beta=0`, decoded at `max_completion_length=64`.

Against the Phase 5 sealed baseline that is **+111%**. Read that number carefully: it is a *recipe*
result. Supervision, reward alignment, and the budget move from 1024 to 64 are bundled into it, and
**no single lever is separable from it**. The sealed split was read once, by explicit authorisation,
and cannot be read again to decompose it (ADR-002).

---

## SFT versus SFT→RL

This is the comparison the project exists to make, and it comes out differently on the two splits.

Each Phase 9 job scored the RL checkpoint **and its own parent SFT** on the same proteins at the same
budget, so the RL step has a paired, same-budget, held-out measurement:

| Seed | RL artifact | SFT arm | RL arm | Paired delta |
|---|---|---:|---:|---:|
| 0 | `bioreasonpro-rl-checkpoint:v6` | 0.17136 | 0.17422 | +0.00286 |
| 1 | `bioreasonpro-rl-checkpoint:v16` | 0.17005 | 0.17265 | +0.00260 |
| 2 | `bioreasonpro-rl-checkpoint:v17` | 0.17063 | 0.18774 | **+0.01712** |
| | | | **mean** | **+0.00753** |

**+0.00753 at t = 1.57 (df = 2) — not established on held-out data.** On validation the same step
measured `+0.02893` at 3.7 sigma. One seed carries the entire sealed delta; the other two average
`+0.0027`.

Three honest qualifications, in both directions:

- This does **not** retract the validation result (ADR-005), whose comparison was RL-vs-RL. It bounds
  how far that result generalises.
- The test is genuinely underpowered at three seeds. "Not established" here means **unresolved, not
  refuted** — a real effect of `+0.0075` would be very hard to detect with df = 2.
- It cannot be resolved by reading the sealed split again. That is the cost of a one-time holdout, paid
  knowingly.

**So the defensible claim is narrower than the headline.** The supervision change carries to held-out
data; the RL step, on this evidence, does not demonstrably.

---

## What replicated, and what that cost

The project's most useful output is the set of conclusions it had to withdraw.

| Claim | Fate |
|---|---|
| Leaf-only supervision | **Replicated**, 3 seeds/arm: `+0.06228`, t = 12.1 — but **13% smaller** than the single-seed estimate |
| Aspect-mean reward | **Replicated**, 3 seeds/arm: `+0.02626` at 6.3 sigma. Wrongly retracted once, then restored |
| RL step on held-out data | **Not established**: `+0.00753`, t = 1.57 |
| `num_generations` 8 → 16 | **Not established**: `+0.01733`, 1.9 sigma; also *less* reproducible (sd 0.0145 vs 0.0069) |
| Ancestor-specificity penalty | **Genuine null**: mechanism confirmed live, metric flat |
| KL anchor `beta=0.04` | **Harmful**: `-0.03117`, t = −6.98 |
| GSPO | **Unreachable**, not tested — the loop has no importance-sampling ratio, so GSPO ≡ GRPO on this path |

Two failures of measurement are worth more than any of the wins:

**The single-seed control, not the treatment, held the error.** Leaf-only supervision was recorded at
+41% from one seed per arm. At three seeds it is +33.4%. The treatment arm reproduced almost exactly
(0.24889 vs 0.24701); the *control* replicated a full point above its recorded value, at twice the
treatment's seed sd. A one-seed control understates its own arm as easily as a one-seed treatment
overstates its own, and it is the less scrutinised of the two.

**Greedy decoding is not bit-reproducible here.** The identical checkpoint evaluated three times scored
`0.17136` / `0.17005` / `0.17063` — sd `0.00066` where zero was expected. Too small to explain anything
above, but it means sub-`0.001` differences are indistinguishable from re-running one arm twice. Cause
undiagnosed (ADR-012).

---

## Evidence chain

Every number above reconstructs backwards through immutable Artifacts. Nothing rests on prose.

**Project.** [`wandb-healthcare/bioreasonpro-senpai`](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai)

**The sealed result** — one aggregate Artifact, built by *reading the run summaries* rather than
recomputing, so it cannot silently disagree with the runs it cites:

```
bioreasonpro-phase9-sealed-evidence:v0        (run cy2f9w9e)
├── bioreasonpro-sft-checkpoint:v4            shared SFT parent of all three seeds
├── bioreasonpro-rl-checkpoint:v6 / v16 / v17 the three evaluated seeds
└── bioreasonpro-phase5-full-holdout:v3/v4/v5 the three per-seed comparison records
```

**Per-seed sealed runs.** [`53xz4re3`](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/53xz4re3)
(job 779) · [`pp1clrgx`](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/pp1clrgx) (780) ·
[`wglc5ivr`](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/wglc5ivr) (781). All tagged
`sealed`; so is the 4-protein plumbing smoke [`gtklbm01`](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/gtklbm01)
(778), because a sealed read that is not on the record is worse than one that is.

**The supervision replication (Phase 8).** Six SFT runs at one revision (`4148839`), artifacts
`sft-checkpoint:v11`–`v16`, swept on val-256 as runs `9crwtepo` / `c4jcvl1g` / `2knfjbu5` (leaf_only)
and `mf66erzp` / `r5qkxmvj` / `vqmp0ogx` (full_closure). The knob smoke pair is `dipscney` / `o6qfg2qu`.

**Walking any evaluation back to its condition.** The evaluated artifact is absent from sweep configs,
so follow the lineage edge rather than reading a field:

```
eval run → rl-checkpoint:vN → producing RL run → condition (config + tags)
```

**Weave.** Traces live under the same project's
[Weave tab](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/weave). RL rollouts are `rollout`
roots with searchable attributes `{stage, rl_algo, importance_sampling_level}`, filterable by
`inputs.group_id` / `inputs.rollout_index` / `attrs.rl_global_step`. Scorers (`ia_f1_scorer`,
`coverage_scorer`, `redundancy_scorer`) appear as child calls of `Evaluation.evaluate` roots.

**Tag vocabulary and the historical backfill.** [`WANDB_TAGS.md`](WANDB_TAGS.md). Note that
`reward_variant` is missing from every RL run before 2026-08-12 — including four runs that are half the
evidence for the aspect-mean result — and survives only as a tag repaired from cluster stdout.

---

## The transferable output

The [method rules in `plan.md`](../plan.md#method-rules) are the part of this project most likely to be
useful elsewhere. Each cost a wrong conclusion that had to be retracted:

1. Sweep the generation budget per arm and compare peaks — a comparison at one shared budget measures
   the budget, not the model. This reversed three separate conclusions.
2. Bracket every peak on both sides.
3. Use the right noise floor, measured per condition — and **budget the seeds before the conditions**.
4. Check `weighted_fmax_n_aspects` before ranking. A `0.4767` that looked like a win was an MF-only mean.
5. Do not select an SFT by its own score if it feeds RL.
6. Verify a knob took effect before trusting the run.
7. Read the in-loop metric as a hint, never as a result.
8. Confirm the code implements the axis before recording a result for it — five axes were *unreachable*
   rather than untried.
9. When bracketing conflicts with comparability, report the peak as bounded rather than fabricating one.
10. Distinguish a genuine null from an unreachable axis.

To which Phases 8 and 9 add two:

11. **Replicate the control, not just the treatment.** The overstatement lived in the single-seed
    baseline, and baselines get less scrutiny than the thing under test.
12. **Verify determinism when it is free.** Scoring one checkpoint twice cost nothing here and exposed a
    noise floor that had been assumed to be zero.

---

## Reproducing it

See [`../RUNBOOK.md`](../RUNBOOK.md). Every element of the recipe is opt-in, so running the launchers
with their defaults reproduces the *original* recipe, not this one.

```bash
# 1. SFT with leaf-only supervision
scripts/submit_slurm.sh slurm/sft.sbatch SENPAI_TARGET_VARIANT=leaf_only

# 2. GRPO from that artifact's receipt, with the aspect-aware reward
scripts/submit_slurm.sh slurm/rl_grpo.sbatch \
  SFT_ARTIFACT_RECEIPT=<runtime_root>/artifacts/sft-<jobid>/wandb_artifact.json \
  SENPAI_TARGET_VARIANT=leaf_only \
  SENPAI_REWARD_VARIANT=aspect_mean \
  RL_NUM_GENERATIONS=8 \
  RL_MAX_STEPS=50 \
  RL_MAX_COMPLETION_LENGTH=64
```

Then evaluate at `max_completion_length=64`, **not** the 1024 default. Expect validation weighted F_max
around `0.276 ± 0.007` across seeds — and expect a single seed to land as high as `0.310`, which is not
a result.
