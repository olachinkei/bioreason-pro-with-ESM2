# bioreason-pro-with-ESM2

`bioreason-pro-with-ESM2` is a reference implementation of a BioReason-Pro-style training workflow:

**supervised fine-tuning (SFT) → reinforcement learning (GRPO) → holdout evaluation.**
Experiments throughout were tracked with W&B Models / W&B Weave.

The repository is designed for people who want to study the workflow itself: how a multimodal
protein-reasoning model is trained, evaluated, and observed. It is not intended to reproduce the
paper's ESM3 checkpoint or headline score.

The public BioReason-Pro work provides the model idea, data, inference components, and evaluation
context, but it did not provide the complete, runnable workflow needed for this reference project:
an end-to-end SFT → GRPO implementation.

Biology is the application. The main subject is the reproducible ML engineering workflow.

## What a completed run of this repository gives you

Concretely, when the phases described in [`report.md`](report.md) are finished you have:

- A license-clean multimodal protein-reasoning model (Qwen3-4B-Thinking + ESM2-650M) you can rebuild
  from scratch, trained by SFT and continued with GRPO.
- A model hand-off chain that cannot silently break: SFT publishes an immutable W&B Artifact, RL
  consumes it through `use_artifact`, and any result reconstructs back to the exact condition that
  produced it.
- A sealed-holdout evaluation that steering never touches.

What you do **not** get is a competitive CAFA score or a reproduction of the paper's result — a
deliberate consequence of the license boundary, not an outstanding task.

## Differences from the BioReason-Pro paper

| Dimension | The paper | Here | Why |
|---|---|---|---|
| Protein encoder | ESM3 | ESM2-650M (MIT) | ESM3 is non-commercial; see License-first design |
| Baseline checkpoint | released BioReason-Pro checkpoint | not used at all | it contains ESM3 weights, so not even as a benchmark |
| Reasoning supervision | GPT-5 generated corpus, 130K+ proteins | **adopted** (ADR-017) | provenance recorded; traces verified to invent nothing (ADR-021) |
| Prompt content | sequence, organism, InterPro, STRING, **PDB structure**, localisation, GO-GPT hypotheses | sequence, organism, InterPro, STRING, localisation, GO-GPT hypotheses | only PDB structure is absent — it feeds ESM3's own encoder in the paper, not the LLM's text prompt |
| **Test split** | **temporal**: trained to Nov 2022, tested on Mar 2023–Feb 2024 annotations with no prior annotation in the target aspect | sealed `bioreason-pro-test-data`; **validation is a random hash split** | the load-bearing gap — see ADR-022 |
| **Model output** | reasoning + **functional summary** + GO terms | reasoning + GO terms | two of the paper's three headline results have no counterpart here |
| Answer targets | as published | authored deterministically from `go_mf`/`go_bp`/`go_cc` | reproducible from pinned label columns |
| RL | as published | GRPO with an IA-weighted F1 proxy reward | reward weights and KL are ours, not the paper's |
| Scale | as published | bounded (5,000 SFT steps; 25–100 RL steps) | fits a reviewable budget |

The most consequential of these is the **test split**. A temporal holdout restricted to proteins with
no prior annotation in the target aspect is designed to defeat annotation transfer; a random split of
an annotated corpus is not. On this repository's validation split a zero-parameter `interpro2go`
lookup scores **0.214** against the trained model's **0.189** — see ADR-022. Any comparison to the
paper's `73.6%` would be comparing different tasks on different encoders, in the easier direction.

The paper reports three headline results; this repository measures one of them. There is no functional
summary and no human-preference study here.

## Current limitations

Honest state of the work, so nobody has to rediscover these.

**Evaluation**

- The sealed holdout has now been read, once, by explicit authorisation (2026-08-14). The recommended
  recipe scores **`0.17820 ± 0.00830`** across three seeds on all 8,630 proteins at a 64-token budget,
  against Phase 5's `0.084657` at 1024. That comparison is **recipe-vs-recipe, not same-budget**, and it
  bundles supervision, reward alignment, and the budget change together — no lever is separable from it.
- **The RL step is not established on held-out data.** Each sealed job scored the RL checkpoint and its
  own parent SFT on the same proteins, giving a paired measurement: `+0.00753`, t = 1.57 (df = 2), with
  one of the three seeds carrying the entire delta. On validation the same step measured `+0.02893` at
  3.7 sigma. The sealed test is underpowered, so this is unresolved rather than refuted — and the sealed
  split is now spent, so it cannot be resolved by reading it again.
- **Greedy decoding is not bit-reproducible on this path.** The identical SFT checkpoint evaluated three
  times gave `0.17136` / `0.17005` / `0.17063` — sd `0.00066` where the answer should have been zero.
  Differences below ~`0.001` between two arms are indistinguishable from re-running one arm twice.
- All steering numbers come from one deterministic 256-protein subset of the 12,280 validation proteins.
  Repeat-measurement spread on it is ~0.004; on the 8,630-protein sealed target the same quantity is
  `0.00066` (Phase 9), so the noise floor is a property of the target and sample size, not a constant.
  Subset-choice error is not separately characterised on either.
- The in-loop 64-protein validation metric disagreed with the 256-protein sweep **six times, in both
  directions**. It is a progress signal, not a result.
- Below roughly 32–48 tokens the metric silently becomes a one- or two-aspect mean and stops being
  comparable, which also means a peak near that boundary cannot always be bracketed from below.

**Statistics**

- Training-seed variance is condition-dependent (sd 0.0022–0.0145). Seven conditions are replicated at
  three seeds; the rest are single-seed and therefore **underpowered rather than refuted**.
- Leaf-only supervision replicated (Phase 8): `+0.06228`, t = 12.1, three seeds per arm. It came in
  **13% smaller than the single-seed estimate** (+41% → +33.4%), and the error was in the *control*:
  `full_closure` replicated at 0.18661 against a recorded `~0.174`, at twice the treatment's seed sd.
  A one-seed control understates its own arm as easily as a one-seed treatment overstates its own.
- `esm_layer` 28 and 30 were only ever measured at an off-peak budget, so they are unmeasured, not
  rejected.

**Model and reward**

- The RL reward is an IA-weighted F1 proxy for the threshold-swept cafaeval F_max. Their correlation was
  never calibrated.
- Reward weights (`lambda_fmt=0.1`, `lambda_len=0.05`) are unpinned placeholders, still marked TODO in
  `bioreason_pro/rewards.py`.
- The multimodal GRPO loop is strictly on-policy with **no importance-sampling ratio**, so GSPO is not
  implementable there — it is out of scope rather than tested.
- `beta` was measured only at 0 and 0.04; at the 50-step operating point the anchor was clearly harmful.
- Multimodal decoding runs one sequence at a time, so generation cost, not training, is what limits
  iteration speed.

**Record keeping**

- `reward_variant` is missing from the config of every RL run before 2026-08-12, including four runs that
  are half the evidence for the project's strongest result. It is repaired via tags — see
  [`docs/WANDB_TAGS.md`](docs/WANDB_TAGS.md) — but the repair had to read cluster stdout that could have
  been deleted.
- The evaluated artifact is absent from sweep-run configs and recoverable only through W&B lineage.
- **Fixed going forward, 2026-08-21:** every run now gets a derived tag and a one-line note at
  `wandb.init()` time (`bioreason_pro/wandb_meta.py`), so a run is orientable from the run list
  without opening the config tab — see [`docs/WANDB_TAGS.md`](docs/WANDB_TAGS.md#every-run-gets-a-derived-tag-and-a-note-the-moment-it-starts).
  This does not retroactively fix the two gaps above; it stops the next one from needing a repair.

## License-first design

The original BioReason-Pro architecture uses ESM3. ESM3 and ESM-C 600M weights are covered by the
EvolutionaryScale Cambrian Non-Commercial License, so they are outside this repository's supported
workflow.

There is no “benchmark-only” exception here:

- the released ESM3 BioReason-Pro checkpoint is not loaded or evaluated;
- ESM3/ESM-C weights, embeddings, outputs, and derived artifacts are not training inputs;
- the protein encoder of record is
  [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D) (MIT);
- the text backbone is
  [`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
  (Apache-2.0); and
- unknown or incompatible asset licenses fail closed.

The machine-readable source of truth is
[`bioreason_pro/approved_assets.json`](bioreason_pro/approved_assets.json). It records the exact
model and dataset revisions, license metadata, provenance status, approved uses, and cache policy.
Training, evaluation, checkpoint reload, and Slurm preflight use that same manifest.

Dataset-card metadata is only the start of the review. SFT, RL, and public-holdout revisions are
pinned down to their Parquet checksums. The supported contract uses only `protein_id`, `sequence`,
and the `go_mf`/`go_bp`/`go_cc` labels. Generated reasoning/final answers, `go_pred`, InterPro/PPI
summaries, and free-text metadata are discarded before prompt construction. The public test dataset
is approved only as a sealed holdout.

This is intentionally a license-contained training flow, not an attempt to reproduce the paper's
generated reasoning corpus or ESM3 result.

## The workflow

### 1. Prepare approved assets

Pin the model and dataset revisions, record their licenses and provenance, and verify that the
selected prompt fields do not contain outputs from an incompatible model. New assets are denied
until a reviewed manifest entry explicitly approves their intended use.

### 2. Run SFT with ESM2

`train.py --stage sft` assembles Qwen + ESM2 fusion and trains the
text adapter and modality projections. A complete checkpoint includes:

- the text-model adapter;
- `projections.pt`; and
- `run_args.json`, including the manifest schema and pinned text/protein model revisions.
- `data_manifest.json`, including every row source, prompt/label field, and split rule.

Saving only the LoRA adapter is insufficient because reloaded random projections make multimodal
evaluation meaningless.

The pinned ESM2 config declares 1,026 positions. The supported fusion contract therefore caps
proteins at 1,024 residues and reserves the remaining two tokens for BOS/EOS. Validate the real
model, run a one-step SFT smoke, and reload its checkpoint on a GPU node:

```bash
uv run --extra texttrain python scripts/verify_esm2_contract.py
uv run --extra texttrain python train.py \
  --stage sft --smoke true --use_multimodal true \
  --model_name Qwen/Qwen3-0.6B \
  --max_steps 1 --gradient_accumulation_steps 1 \
  --max_completion_length 8 --eval_subset_size 1 \
  --lora_r 16 --lora_alpha 32 \
  --output_dir /mnt/data/$USER/BioReason-Pro/artifacts/sft-smoke \
  --wandb_name sft-smoke
uv run --extra texttrain python scripts/verify_sft_checkpoint.py \
  --checkpoint /mnt/data/$USER/BioReason-Pro/artifacts/sft-smoke
```

The training command fails on non-finite loss or gradients and writes the adapter,
`projections.pt`, `run_args.json`, and `data_manifest.json`. The reload command writes
`reload_verification.json` after a finite fused forward and protein-aware generation. Training also
publishes the complete directory as `bioreasonpro-sft-checkpoint:vN`, waits for W&B to finalize the
version, and writes its immutable reference to `wandb_artifact.json`.

The custom streaming loops always have a finite default. One logical streaming epoch is 500
optimizer steps, so the default 10 epochs stop after 5,000 steps and then save, evaluate, and log
the W&B Artifact. `data_subset_frac` scales that bound, `SENPAI_MAX_EPOCHS` can lower it, and an
explicit positive `--max_steps` takes precedence. Values of `0` or less than `-1` are rejected so a
mistyped bound cannot silently create an unbounded GPU job.

### 3. Continue with GRPO

`train.py --stage rl` starts from the approved SFT result, merges the SFT adapter, creates a fresh RL
adapter, generates grouped completions, scores them, and applies the policy update. W&B records
training metrics; Weave stores inspectable rollout traces.

The hand-off is a W&B Artifact contract, not a local path contract. The RL run accepts only an
immutable `entity/project/artifact:vN` reference and calls the active run's `use_artifact` before
downloading or loading it. This creates the lineage edge `SFT artifact → RL run`. The run restores
the SFT LoRA **and** its learned protein projections, creates a fresh RL LoRA, and publishes the
complete output as an RL model artifact, creating `RL run → RL artifact`. The RL artifact also
bundles its exact SFT parent so standalone reload remains deterministic.

A bounded two-generation smoke is available after an SFT checkpoint succeeds:

```bash
uv run --extra texttrain python train.py \
  --stage rl --rl_algo grpo --smoke true --use_multimodal true \
  --model_name Qwen/Qwen3-4B-Thinking-2507 \
  --sft_artifact wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:vN \
  --max_steps 1 --num_generations 2 --max_completion_length 8 \
  --eval_subset_size 1 --lora_r 16 --lora_alpha 32 --beta 0.0 \
  --output_dir /mnt/data/$USER/BioReason-Pro/artifacts/rl-grpo \
  --wandb_name rl-grpo-smoke
```

W&B receives total/component rewards, reward variance, KL, completion length, gradient norm, and
sample rollouts. Weave records each completion with protein ID, group, rollout index, step, reward
components, and checkpoint lineage. The comparison run also calls `use_artifact` for both immutable
model versions before evaluating them:

```bash
uv run --extra texttrain python scripts/compare_rl_checkpoints.py \
  --sft-artifact wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:vN \
  --rl-artifact wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:vN \
  --output outputs/rl-comparison.json
```

After the smoke succeeds, `slurm/rl_grpo.sbatch` runs the bounded eight-GPU reference
experiment. Its defaults are 100 optimizer steps, eight generations per prompt, 256 generated
tokens, and a shared 64-example evaluation slice. Each value can be overridden explicitly with an
`RL_*` environment variable, while the 24-hour Slurm limit and 18-hour training deadline leave
time to save, reload, finalize the RL Artifact, and run the comparison. Artifact downloads and
job-specific caches are removed after finalization; the output checkpoint, receipt, comparison,
and runtime log are retained.

#### Completed reference run

- [GRPO run](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/41fsrvnj)
- [paired comparison run](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/t12uylu5)

### 4. Evaluate on an available holdout

The preferred paper-adjacent CAFA5 dataset is access-gated on Hugging Face. It is optional and stays
disabled until access, schema, license, and provenance are verified.

The no-approval sample path uses the public
[`wanglab/bioreason-pro-test-data`](https://huggingface.co/datasets/wanglab/bioreason-pro-test-data)
split as a sealed holdout:

```bash
python eval.py \
  --split test \
  --target bioreason_pro_test \
  --checkpoint <complete-esm2-checkpoint>
```

For a quick plumbing check, pass an explicit `--subset_size`. The full test split remains the
default for final evaluation. The test set must not be used for training, model selection, or
hyperparameter tuning.

The Artifact-to-Artifact comparison is distributed across the GPUs allocated by Slurm.
It fails unless the joined shards contain exactly the requested subset—or all 8,630 proteins for
the default full run—with unique protein IDs and identical ground truth for both models. A small
two-GPU plumbing check can use an explicit subset:

```bash
sbatch --gres=gpu:h100:2 --cpus-per-task=24 --mem=450G --time=02:00:00 \
  --export=ALL,SOURCE_REVISION=<commit>,SOURCE_BUNDLE=<bundle>,EVAL_WORLD_SIZE=2,EVAL_SUBSET_SIZE=4,SFT_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:v2,RL_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:v2 \
  slurm/eval_full_holdout.sbatch
```

After that succeeds, omit both smoke overrides to evaluate the complete sealed holdout on eight
H100 GPUs. The single W&B evaluation run calls `use_artifact` for both immutable models before any
rank loads them and publishes the result as a versioned evaluation Artifact:

```bash
sbatch \
  --export=ALL,SOURCE_REVISION=<commit>,SOURCE_BUNDLE=<bundle>,SFT_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:v2,RL_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:v2 \
  slurm/eval_full_holdout.sbatch
```

#### Completed full-holdout run

[W&B comparison run](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai/runs/hhn45qgc)

### 5. Compare and visualize

The reference comparison is:

| Run | What it demonstrates |
|---|---|
| ESM2 SFT | a commercially usable, reloadable supervised baseline |
| ESM2 SFT → GRPO | reinforcement learning from that same approved baseline |
| Optional text-only SFT | an ablation for the protein modality |

The important output is the traceable workflow: code revision, asset revisions, configuration,
metrics, checkpoint artifacts, system telemetry, and rollout evidence in W&B/Weave. Improvement is
desirable, but the sample remains useful even when RL does not beat SFT.

## Repository layout

| Path | Purpose |
|---|---|
| `docs/reference/PAPER.md` | the paper this repository reimplements: its setup, and every way ours differs |
| `docs/FINAL_REPORT.md` | the result: SFT vs SFT→RL, the evidence chain, and what did not replicate |
| `docs/ROLLOUT_ANALYSIS.md` | what 472 stored rollouts show about why each mechanism did or did not work |
| `scripts/analyze_rollouts.py` | re-derives every number in that analysis from the rollout tables |
| `report.md` | plain-language writeup of the headline SFT-vs-RL result |
| `docs/WANDB_TAGS.md` | W&B run tag rules and controlled vocabulary |
| `scripts/apply_run_tags.py` | applies the tag table idempotently (dry-run by default) |
| `bioreason_pro/approved_assets.json` | pinned asset allowlist and approved-use policy |
| `train.py` | SFT and RL entrypoint, checkpointing, and model reload |
| `eval.py` | protected holdout scorer entrypoint |
| `eval_targets/` | access-aware dataset target registry |
| `bioreason_pro/` | fusion, encoder, reward, GO, and tracing components |
| `slurm/` | CoreWeave/SUNK launch scripts |
| `scripts/submit_slurm.sh` | stages the source bundle for a revision, then submits a launcher |
| `slurm/sft_smoke.sbatch` | one-GPU SFT artifact producer for the lineage smoke |
| `slurm/sft.sbatch` | full eight-GPU approved ESM2 SFT launcher |
| `slurm/rl_grpo_smoke.sbatch` | bounded SFT→GRPO continuation and comparison |
| `slurm/rl_grpo.sbatch` | eight-GPU bounded GRPO run, reload check, Artifact publication, and comparison |
| `slurm/eval_full_holdout.sbatch` | distributed immutable-model comparison over a smoke subset or the full sealed holdout |
| `tests/` | offline and pre-GPU contract tests |
| `RUNBOOK.md` | launch, monitoring, recovery, and teardown guidance |

## Quickstart

```bash
uv sync
uv run pytest -q
python scripts/fetch_approved_reference_data.py
```

Training requires the appropriate model/data access and GPU environment. Start with the offline
tests and the checked-in synthetic one-batch fixture. The fetch command installs the exact
CC-BY-4.0 CAFA evaluation ontology/IA pair after verifying archive and file checksums; it will not
replace mismatched local files unless `--force` is explicitly supplied.

### Cluster / GPU environment

Every `slurm/*.sbatch` launcher in this repo was built for and tested on a **CoreWeave SUNK** Slurm
cluster (H100 partition, Pyxis/Enroot containers, a shared `/mnt/data` + `/mnt/home` filesystem). On
any other Slurm cluster, submit through `scripts/submit_slurm.sh` and set:

- `SENPAI_CLUSTER_HOST` — your login host (`user@login.example.com`); required, no default.
- `SENPAI_REMOTE_ROOT` — a writable directory on your cluster's shared filesystem; required, no default.
- `SENPAI_SBATCH_ARGS` (optional) — overrides sbatch directives baked into the launchers, e.g.
  `--partition=a100 --gres=gpu:a100:8`, since command-line sbatch flags take precedence over a
  script's own `#SBATCH` lines.
- `SENPAI_CONTAINER_MOUNTS` (optional) — overrides the `--container-mounts` value if your cluster's
  shared paths or Pyxis/Enroot mount syntax differ from `/mnt/data:/mnt/data,/mnt/home:/mnt/home`.

```bash
SENPAI_CLUSTER_HOST=user@login.example.com \
SENPAI_REMOTE_ROOT=/mnt/data/user/BioReason-Pro \
scripts/submit_slurm.sh slurm/sft_smoke.sbatch
```

## Project status

The SFT, multimodal generation, GRPO, checkpoint, evaluation-target, W&B, Weave, and Slurm building
blocks exist, and the model/checkpoint/data boundaries fail closed around pinned
Apache-2.0/MIT/CC-BY-4.0 assets. SFT, GRPO, and the full public-holdout comparison have each
completed and published their own versioned, reloadable W&B Artifact (see the runs linked above). A
later training/decoding-condition search closed on every reachable axis without re-touching the
sealed holdout.

For the headline result in plain language, see [`report.md`](report.md).
