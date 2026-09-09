# What 472 rollouts say about a result that only half replicated

A closing report. The supervision fix held up at three seeds and twelve sigma; the reinforcement-learning
step did not carry to held-out data. This reads the stored RL rollouts to see *why* both happened — and
finds one thing nobody had claimed.

Numbers here are re-derivable, not transcribed:

```bash
uv run python scripts/analyze_rollouts.py
```

Companion documents: the result and its evidence chain is [`FINAL_REPORT.md`](FINAL_REPORT.md); the
decisions are [`../plan.md`](../plan.md).

---

## Summary

| | |
|---|---:|
| Sealed holdout, recommended recipe, 3 seeds | **0.17820 ± 0.00830** |
| Supervision leg (`leaf_only`), 3 seeds/arm | **+0.06228**, t = 12.1 |
| RL leg on held-out data, paired, 3 seeds | **+0.00753**, t = 1.57 — *not established* |
| Rollouts examined | **472**, 8 conditions |

**Phase 8** replicated the supervision leg: six SFT runs at one revision, three seeds each, every peak
bracketed on both sides. It confirmed the claim and shrank it — +41% became +33.4% — and the error was
in the *control*, which had been measured at a single seed.

**Phase 9** spent the project's one authorised sealed read: three seeds at the validation-tuned 64-token
budget over all 8,630 held-out proteins. The recipe more than doubled the prior sealed baseline, but the
RL step on its own did not separate from seed noise.

**Phase 10** published the workflow. This analysis is new work on top of it.

---

## Population and its shape

472 rollouts from the `rl/rollout_samples` tables of eight RL runs, at optimizer steps 1–100.

**One table is one GRPO group: `num_generations` completions of a single protein.** That shape is the
easiest thing to get wrong here. Each step logs a *different* protein, so any trend read across steps is
confounded by protein identity — a first pass at this analysis produced a clean-looking "aspect drift
over training" that was measuring which proteins happened to come up. The comparison below is paired on
`(step, protein)` instead, which is available because both arms consume the same data order and
therefore see the same protein at the same step.

---

## Finding 1 — the thinking model never thinks

The backbone is `Qwen/Qwen3-4B-Thinking-2507`. In **472 of 472 rollouts** the reasoning block is opened
and closed empty:

```
<think>

</think>

MF: GO:0005515
BP: GO:0006511
CC: GO:0005739, GO:0032040, GO:0055038, GO:01⌐ cut at 64 tokens
```
<sub>`Q9BRX2` · aspect_mean / leaf_only · step 50 · reward 0.3097 · F_max 0.2097</sub>

Not once across eight conditions, six training steps and three reward variants did it spend a token
reasoning. This reframes ADR-006: at budget 64 the budget was never being split between reasoning and
answer. A thinking model was being paid for and not used.

## Finding 2 — every answer is cut off mid-identifier

**466 of 472 rollouts (98.7%)** end partway through a GO identifier — `GO:01`, `GO:00`, `GO:012025`. The
model never chooses to stop; the budget stops it.

So the term counts below are not what the model wanted to emit, they are what fits. The informative
question is not *how many* terms but **which aspect the budget gets spent on** — which is exactly where
the reward variants diverge.

## Finding 3 — the shipped baseline spent its budget naming the root of the ontology

The original `full_closure` recipe on protein `B6K141`: 24 terms, F_max `0.0584`. Its molecular-function
list is one ancestor chain walked from the top.

| GO term | Name | |
|---|---|---|
| `GO:0003674` | molecular_function | the root of the entire ontology |
| `GO:0005488` | binding | |
| `GO:0005515` | protein binding | |
| `GO:0042802` | identical protein binding | |
| `GO:0046983` | protein dimerization activity | |
| `GO:0042803` | protein homodimerization activity | the only term carrying information |

Six terms making one claim: *this protein homodimerizes*. `cafaeval` propagates predictions up the
hierarchy, so the five ancestors are already implied by the sixth and earn nothing.

Measured against the real ontology with the scorer's own ancestor closure, **64.9%** of the terms this
arm emits are proper ancestors of another term it emitted. Under leaf-only supervision that falls to
**2.5%** in the recommended arm and stays within **0–6.2%** across every leaf-only condition.

That is ADR-004's +33.4% visible as text rather than as a table cell.

## Finding 4 — the union reward really does pile into the cheapest aspect

ADR-005 claimed that scoring one metric over the union of all three aspects lets the optimiser push
whichever aspect is cheapest, instead of raising the per-aspect mean the evaluation actually uses. The
rollouts test that directly, paired on protein and step, 16 rollouts per arm per cell:

| Step | Protein | union MF/BP/CC | aspect_mean MF/BP/CC | CC share, union | CC share, aspect_mean |
|---:|---|---|---|---:|---:|
| 1 | `Q13542` | 0.81 / 2.06 / 2.12 | 0.81 / 2.06 / 2.12 | 42% | 42% |
| 10 | `Q5ADW3` | 1.31 / 1.81 / 1.88 | 1.31 / 1.88 / 1.81 | 38% | 36% |
| 20 | `Q8IYD1` | 1.31 / 1.81 / 1.88 | 1.31 / 2.00 / 1.69 | 38% | 34% |
| 30 | `A0A482NB13` | 0.38 / 1.25 / 3.38 | 1.12 / 1.56 / 2.31 | **68%** | 46% |
| 40 | `O00566` | 0.94 / 0.94 / 3.12 | 1.25 / 1.50 / 2.25 | **62%** | 45% |
| 50 | `Q9BRX2` | 1.00 / 1.12 / 2.88 | 1.31 / 1.75 / 1.94 | **57%** | 39% |
| | | | **mean** | **50.8%** | **40.4%** |

Paired difference **+10.4 points, t = 2.62 (df = 5)**.

Step 1 is identical by construction — same checkpoint, same prompt, before either reward has applied a
gradient. The arms separate from step 30, in the predicted direction. On `A0A482NB13` the union arm
spends 3.38 terms on cellular component against the aspect-mean arm's 2.31, and pays for it in molecular
function: 0.38 against 1.12. Same protein, same step, opposite allocation.

Same protein, two rewards, at step 50:

```
union         MF: GO:0003723
              BP: GO:0000398
              CC: GO:0070062, GO:0070064, GO:0070065, GO:00⌐

aspect_mean   MF: GO:0005515
              BP: GO:0006511
              CC: GO:0005739, GO:0032040, GO:0055038, GO:01⌐
```

## Finding 5 — everything in the reward except F_max was a constant

This one nobody had claimed, and it only appears by looking at the raw numbers.

Subtracting each rollout's F_max from its total reward leaves whatever the auxiliary terms contributed.
Across all 472 rollouts that remainder is **identically 0.1 in 459 of them (97.2%)**. The other 13 sit at
0.08 — and all 13 are from the `aspect_mean_specific` arm, where the gap is that variant's
deliberately enabled `lambda_spec` redundancy penalty.

| reward − F_max | Rollouts | Share | Meaning |
|---:|---:|---:|---|
| +0.100 | 459 | 97.2% | format bonus fully earned — a constant |
| +0.080 | 13 | 2.8% | all from `aspect_mean_specific`; its `lambda_spec` penalty, working as designed |

So in **every arm that shipped**, `reward − F_max` is a constant with no exceptions at all. The reason
is structural: `r_format` scores `0.5 × (a <think> block exists) + 0.5 × (at least one GO id exists)`,
and both conditions hold for **all 472** completions. It is not that the term rarely varies — at this
operating point it *cannot*.

**A constant reward component is worse than no reward component.** GRPO centres advantages *within* each
group of rollouts, so a term returning the same value for every member of the group contributes exactly
zero to the advantage after centring. For 97% of groups this bonus is a constant offset that cannot
teach anything. It varies only when the model drops a whole aspect line — 13 times in 472 — so it is not
shaping behaviour at the margin, it is a rare all-or-nothing penalty.

Every gradient the RL step actually followed came from F_max.

The README already flagged `lambda_fmt=0.1` and `lambda_len=0.05` as unpinned placeholders still marked
TODO in `bioreason_pro/rewards.py`. This says something sharper: at the operating point actually
shipped, they were not placeholder *values* — they were inert.

## Finding 6 — the learning signal is thin, and occasionally absent

GRPO learns only from disagreement inside a rollout group. Where all generations score alike, the step
teaches nothing.

Within-group reward spread starts at 0.16–0.21 at step 1 and settles to 0.02–0.08 by step 50: the group
converges and the gradient shrinks with it. In the shipped baseline one step recorded **exactly zero
spread** — eight identical scores, one wasted optimiser step.

With 50 steps of 8 rollouts, a thin and decaying signal plus a single shared SFT starting point is a
plausible mechanic for what the sealed numbers showed: an RL gain that is real on the distribution it
was tuned on and does not survive the move to held-out proteins. **That is a hypothesis these rollouts
make available, not something this data settles.**

---

## What holds

- **The recipe more than doubles the sealed score** — `0.084657` → `0.17820`. A recipe result:
  supervision, reward alignment and the 1024→64 budget move are bundled into it, and the sealed split
  cannot be read again to separate them.
- **The supervision leg carries.** Replicated at t = 12.1, and visible here as ancestor redundancy
  collapsing from 64.9% to 2.5%.
- **The reward-alignment mechanism is real.** The union reward demonstrably shifts output toward the
  cheapest aspect, paired and protein-controlled.
- **The RL leg is unresolved on held-out data.** +0.00753 at t = 1.57, carried by one seed of three.
  Underpowered rather than refuted — and no longer resolvable by the route that would have settled it.

Three questions are left open and labelled open: whether the RL step carries beyond validation
(ADR-011), why greedy decoding fails to reproduce exactly (ADR-012), and whether an RL run whose
auxiliary reward is inert and whose group variance decays to 0.02 has enough signal to justify its cost
(ADR-013).

---

## Provenance

Rollouts come from the `rl/rollout_samples` tables of the eight runs listed in
`scripts/analyze_rollouts.py`; the run ids are pinned there so the population being described is
auditable rather than "whatever the API returned today". Redundancy is measured against
`data/go-basic.obo` using `bioreason_pro.go_obo.load_go_ancestors` — the same closure the scorer uses.

Rollout groups are 8–16 generations of a **single** protein per step, so cross-step comparisons are
confounded by protein identity and are not made here. Aspect-share figures are means over 16 rollouts
per cell.
