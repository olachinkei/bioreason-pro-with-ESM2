# bioreason-pro-with-ESM2

Protein-function prediction with **ESM2 + Qwen3**, trained through
**SFT → GRPO → holdout evaluation**. W&B tracks experiments and model artifacts;
Weave records rollout traces.

Based on [BioReason-Pro](https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1)
([upstream code](https://github.com/bowang-lab/BioReason-Pro)), with ESM2 replacing ESM3.

| Component | Model |
|---|---|
| Protein encoder | `facebook/esm2_t33_650M_UR50D` (MIT) |
| Text backbone | `Qwen/Qwen3-4B-Thinking-2507` (Apache-2.0) |

Model and dataset revisions, licenses, and approved uses are pinned in
[`approved_assets.json`](bioreason_pro/approved_assets.json).

## Setup

```bash
uv sync
uv run python scripts/fetch_approved_reference_data.py
uv run pytest -q
```

Configure credentials using [`.env.example`](.env.example). Keep `.env` private.

## Train

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

## Evaluate

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

## Monitor

Run on the cluster login node:

```bash
squeue -u "$USER"
tail -f /mnt/data/$USER/BioReason-Pro/runtime_logs/*.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

Inspect metrics, checkpoints, and comparisons in
[W&B](https://wandb.ai/wandb-healthcare/bioreasonpro-senpai), and rollout traces in Weave.
After resolving a failed job's cause, resubmit with the same source revision and artifact inputs.
