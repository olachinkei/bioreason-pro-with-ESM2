# W&B run tags: rules and vocabulary

A run should be orientable from the W&B run list alone — what stage it is, what it varied, what
it's an evaluation of — without opening the config tab or reading a cluster log.

## The one rule that decides whether a tag should exist

**A tag must encode something that cannot be derived from the run config plus artifact lineage.**
Most of what an experiment varied (`num_generations`, `max_steps`, `beta`, `esm_layer`, `lora_r`,
`target_variant`, `reward_variant`) is already in the config — duplicating it as a tag creates two
sources of truth that drift. Tags earn their place in two situations only:

1. **Judgements** — whether a result is the current best, superseded, a genuine null, or unrankable.
   No config field can hold a conclusion about a run.
2. **Repairs** — a condition the run actually used but its config never recorded, because the
   logging code for that field landed later. Backfilled from cluster stdout where recoverable.

## Every run gets a derived tag and a note automatically

Every `wandb.init()` call passes `tags=`/`notes=` computed by `bioreason_pro.wandb_meta`:

- **Stage tag** — `sft` / `rl` / `eval` / `smoke`. `smoke` marks a plumbing run (tiny step count or
  a synthetic fixture); smoke runs must never appear in a results table.
- **Note** — one line assembled from a fixed, ordered list of config keys (e.g. `RL ·
  target=leaf_only_reasoned · reward=aspect_mean_reasoned · seed=1`), so it can never drift out of
  sync with the config it's derived from.

A new `wandb.init()` call site should import `derived_tags`/`build_note` from `wandb_meta` rather
than write its own tag/note logic.

## Vocabulary

### Split — exactly one on every `eval` run

| Tag | Meaning |
|---|---|
| `val` | Steering split. Free to use as often as needed. |
| `sealed` | The public holdout. Reading it is a deliberate, one-time, separately authorised act. |

### Standing — zero or one

| Tag | Meaning |
|---|---|
| `recommended` | Member of the currently recommended condition's replicate set. Move it when a better condition is found; don't accumulate it. |
| `superseded` | Was quoted as a result and has since been beaten or retracted. |
| `null-result` | Ran correctly, the mechanism was confirmed live, and the metric did not move. |
| `not-comparable` | `weighted_fmax_n_aspects < 3` — the metric averaged fewer than three aspects, so it must never be ranked against a three-aspect mean. |
| `broken-config` | The run executed but the intended condition never took effect. Distinct from `null-result`: nothing was actually tested. |

### Replication — as applicable

`n3` marks a member of a three-seed set; `train_seed` in the config identifies which one.

## Applying tags

`scripts/apply_run_tags.py` holds judgement/repair tags as a declarative table and applies them
idempotently. Dry-run by default; pass `--apply` to write.

```bash
uv run python scripts/apply_run_tags.py            # show the diff
uv run python scripts/apply_run_tags.py --apply    # write it
```

A tag added by hand in the W&B UI is allowed but will be overwritten the next time the script runs
against that run — put it in the table instead.

## Practical guidance

- Tag a run's judgement (recommended/superseded/null-result) when it finishes, not later.
- When a result is beaten, move `recommended` in the same commit that records the new number.
- Never rank a `not-comparable` run against a three-aspect one.
- Prefer fixing the config over adding a tag — a tag is a repair, a recorded field is the real fix.
