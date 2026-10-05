# bioreason-pro-with-ESM2

An independent implementation of **BioReason-Pro** for exploring how to train a model that
reasons about protein function. It pairs **ESM2 with Qwen3** and includes supervised fine-tuning
(SFT), reinforcement learning (GRPO), and evaluation, with experiment tracking in W&B and
rollout tracing in Weave.

## Background

[BioReason-Pro](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1) integrates
protein sequence, structure, domains, and interaction context to predict protein function while
generating biological reasoning. Its ability to make the reasoning behind a prediction inspectable
motivated this project.

The official
[`train_protein_llm.py`](https://github.com/bowang-lab/BioReason-Pro/blob/main/train_protein_llm.py)
provides supervised training, but an end-to-end implementation extending through reinforcement
learning was not available in the upstream materials reviewed for this project. This repository
therefore implements an SFT → GRPO → evaluation workflow to explore how reinforcement learning
can be applied to protein-function reasoning.

## Implementation

| Component | Configuration used here |
|---|---|
| Protein encoder | `facebook/esm2_t33_650M_UR50D` (MIT); amino-acid sequence input |
| Text backbone | `Qwen/Qwen3-4B-Thinking-2507` (Apache-2.0) |
| Supervised training | LoRA fine-tuning with learned modality projections |
| Reinforcement learning | GRPO-style on-policy updates with group-relative rewards; GSPO is not used in the multimodal pipeline |
| Configuration below | Leaf-only GO targets, aspect-aware reward, 8 rollouts per prompt, 50 RL steps, no KL penalty |
| Tracking | W&B model artifacts and metrics; Weave rollout traces |

[ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) was selected for its MIT license,
which permits commercial use subject to its terms. This choice was motivated by the
[Cambrian Non-Commercial License](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)
covering the ESM3 release considered at the start of the project. The
[current ESM3 model card](https://huggingface.co/biohub/esm3-sm-open-v1) now lists MIT.

This implementation uses ESM2 instead of ESM3 to address the licensing constraints considered
at the start of the project, and uses GRPO instead of GSPO to keep the RL implementation simple.
These choices make it an adaptation of BioReason-Pro rather than an exact reproduction of the paper.

Model and dataset revisions, licenses, and approved uses are pinned in
[`approved_assets.json`](bioreason_pro/approved_assets.json).

Corrections to license interpretation, implementation, or differences from the paper are welcome
via [GitHub Issues](https://github.com/olachinkei/bioreason-pro-with-ESM2/issues).

## Setup

```bash
uv sync
uv run python scripts/fetch_approved_reference_data.py
uv run pytest -q
```

Configure credentials using [`.env.example`](.env.example). Keep `.env` private.

## How to use

### Train

Run the one-GPU smoke test first:

```bash
scripts/submit_slurm.sh slurm/sft_smoke.sbatch
```

Then train with leaf-only supervision:

```bash
scripts/submit_slurm.sh slurm/sft.sbatch BIOREASON_PRO_TARGET_VARIANT=leaf_only
```

After SFT finishes, pass its artifact receipt to GRPO. This configuration uses the aspect-aware
reward, eight rollouts per prompt, and no KL penalty:

```bash
scripts/submit_slurm.sh slurm/rl_grpo.sbatch \
  SFT_ARTIFACT_RECEIPT=<runtime_root>/artifacts/sft-<jobid>/wandb_artifact.json \
  BIOREASON_PRO_TARGET_VARIANT=leaf_only \
  BIOREASON_PRO_REWARD_VARIANT=aspect_mean \
  RL_NUM_GENERATIONS=8 \
  RL_BETA=0 \
  RL_MAX_STEPS=50 \
  RL_MAX_COMPLETION_LENGTH=64
```

For an RL smoke test, use `slurm/rl_grpo_smoke.sbatch` with the SFT smoke receipt.
Full training launchers use eight H100 GPUs. Each stage saves and reload-verifies a complete
checkpoint, then publishes a versioned W&B Artifact: **SFT artifact → RL run → RL artifact**.

### Evaluate

Compare immutable SFT and RL artifact versions at the same 64-token generation budget.
Replace `vN` with each artifact's actual version:

```bash
scripts/submit_slurm.sh slurm/eval_full_holdout.sbatch \
  SFT_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:vN \
  RL_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:vN \
  EVAL_MAX_COMPLETION_LENGTH=64
```

The launcher scores all 8,630 public holdout proteins and publishes a W&B evaluation Artifact.
For a small pipeline check, add `EVAL_SUBSET_SIZE=4`.
Use validation data for model selection; keep the holdout out of training and hyperparameter tuning.

### Monitor

Run on the cluster login node:

```bash
squeue -u "$USER"
tail -f /mnt/data/$USER/BioReason-Pro/runtime_logs/*.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

Inspect metrics, checkpoints, and comparisons in
[W&B](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai), and rollout traces in Weave.
After resolving a failed job's cause, resubmit with the same source revision and artifact inputs.
