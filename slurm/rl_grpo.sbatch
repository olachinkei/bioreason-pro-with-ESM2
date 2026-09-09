#!/bin/bash
# Bounded eight-GPU GRPO continuation from a finalized Phase 3 W&B Artifact.
# Submit with:
# sbatch --export=ALL,SOURCE_REVISION=<40-character-commit>,SOURCE_BUNDLE=<bundle>,SFT_ARTIFACT_RECEIPT=<receipt> \
#   slurm/rl_grpo.sbatch
#SBATCH --job-name=bioreason-rl-grpo
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G
# ADR-038 follow-up: job 1735 (proteins_per_step=4, group_mean_global_std) advanced ~18 steps within
# its 18h internal budget (~1h/step) before gracefully stopping -- far short of a 200-step run. At
# max_steps=100 (this file's own default) that pace needs ~100h of training alone; the partition
# itself has MaxTime=UNLIMITED (confirmed via `scontrol show partition h100`), so this 24h ceiling was
# always self-imposed, not a cluster limit. Extended to 7 days to fit 100 steps with real headroom.
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/rl-grpo-%j.out
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_REVISION:?submit with the exact 40-character Git commit to run}"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a lowercase 40-character Git commit" >&2
  exit 2
fi
: "${SFT_ARTIFACT_RECEIPT:?submit with the finalized Phase 3 wandb_artifact.json path}"
export BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"

runtime_root="${BIOREASON_RUNTIME_ROOT:-/mnt/data/$USER/BioReason-Pro}"
mkdir -p "$runtime_root"
source_dir=""
if [[ -z "${REPO_DIR:-}" ]]; then
  source_dir="$runtime_root/source-rl-grpo-$SLURM_JOB_ID"
  git init -q "$source_dir"
  if [[ -n "${SOURCE_BUNDLE:-}" ]]; then
    [[ -r "$SOURCE_BUNDLE" ]] || {
      echo "SOURCE_BUNDLE is not readable: $SOURCE_BUNDLE" >&2
      exit 2
    }
    git -C "$source_dir" fetch -q "$SOURCE_BUNDLE" "$SOURCE_REVISION"
  else
    source_repository_url="${SOURCE_REPOSITORY_URL:-https://github.com/wandb/bioreason-pro-with-ESM2.git}"
    git -C "$source_dir" remote add origin "$source_repository_url"
    git -C "$source_dir" fetch -q --depth 1 origin "$SOURCE_REVISION"
  fi
  git -C "$source_dir" checkout -q --detach FETCH_HEAD
  export REPO_DIR="$source_dir"
fi
actual_revision="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$actual_revision" != "$SOURCE_REVISION" ]]; then
  echo "REPO_DIR is at $actual_revision, expected $SOURCE_REVISION" >&2
  exit 2
fi

export HF_HOME="$runtime_root/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$runtime_root/cache/xdg"
# ADR-036/037 incident: wandb's artifact-staging path follows XDG_DATA_HOME (default ~/.local/share,
# on the container's small local root filesystem) independently of XDG_CACHE_HOME/WANDB_DIR -- job 1692
# completed training and val, then failed only at artifact upload with "disk quota exceeded" writing
# to /root/.local/share/wandb/artifacts/staging. Redirect it to the same ample shared storage.
export XDG_DATA_HOME="$runtime_root/cache/xdg-data"
export TRITON_CACHE_DIR="$runtime_root/cache/triton-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/cache/torch-inductor-$SLURM_JOB_ID"
export WANDB_DIR="$runtime_root/wandb"
export WEAVE_SERVER_CACHE_DIR="$WANDB_DIR/weave_server_cache"
export BIOREASON_REFERENCE_DATA_ROOT="${BIOREASON_REFERENCE_DATA_ROOT:-$runtime_root/reference_data}"
export BIOREASON_ARTIFACT_ROOT="$runtime_root/artifact_inputs/rl-grpo-$SLURM_JOB_ID"
export BIOREASON_JOB_TMP="$runtime_root/tmp/rl-grpo-$SLURM_JOB_ID"
export RL_OUTPUT="${RL_OUTPUT:-$runtime_root/artifacts/rl-grpo-$SLURM_JOB_ID}"
export RL_MODEL_NAME="${RL_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
export RL_MAX_STEPS="${RL_MAX_STEPS:-100}"
export RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-8}"
# ADR-036: the corpus's own reasoning traces run to p99≈1900 tok / max≈2860 tok (measured); 256 (the
# old default here) and even the 1536 Phase-6 submitted value left real truncation on the table
# (val_primary/answer_truncated=0.094 at 1536). 3072 covers p99 with headroom for the answer.
export RL_MAX_COMPLETION_LENGTH="${RL_MAX_COMPLETION_LENGTH:-3072}"
# ADR-037: batch RL_PROTEINS_PER_STEP proteins x RL_NUM_GENERATIONS rollouts per optimizer
# step (default 1 = today's exact single-protein-per-step behaviour). RL_ADVANTAGE_NORMALIZATION
# is "group_mean" (Dr.GRPO, no std division — the pre-ADR-037 default) or "group_mean_global_std"
# (paper's own Eq. 4.19-4.20: per-protein group mean, divided by one std pooled across the batch).
export RL_PROTEINS_PER_STEP="${RL_PROTEINS_PER_STEP:-1}"
export RL_ADVANTAGE_NORMALIZATION="${RL_ADVANTAGE_NORMALIZATION:-group_mean}"
export RL_EVAL_SUBSET_SIZE="${RL_EVAL_SUBSET_SIZE:-64}"
export RL_COMPARISON_SUBSET_SIZE="${RL_COMPARISON_SUBSET_SIZE:-64}"
# val by default: this step runs on every GRPO invocation, so defaulting to the sealed split would
# spend the holdout once per iteration. Set to "test" only for a one-off publication measurement.
export RL_COMPARISON_SPLIT="${RL_COMPARISON_SPLIT:-val}"
export RL_BETA="${RL_BETA:-0.0}"
# Must equal the esm_layer of the SFT artifact being continued: train.py rejects a mismatch
# ("SFT artifact ESM layer N does not match requested M") rather than quietly building the wrong
# model. Defaults to 33, the shipped layer.
export RL_ESM_LAYER="${RL_ESM_LAYER:-33}"
# The RL adapter rank. Note the asymmetry with Phase 3, which trains at r=128: RL has run at r=16
# throughout, so whether the RL leg is capacity-limited has never been tested. alpha conventionally
# tracks 2r.
export RL_LORA_R="${RL_LORA_R:-16}"
export RL_LORA_ALPHA="${RL_LORA_ALPHA:-32}"
# grpo (token-level importance sampling, paper-faithful) or gspo (sequence-level). train.py has
# supported both all along, but this launcher hardcoded grpo, so plan.md's "optional follow-up" was
# in fact unreachable.
export RL_ALGO="${RL_ALGO:-grpo}"
# Stochastic-training seed only; the data split stays on --seed. -1 = shipped behaviour.
export RL_TRAIN_SEED="${RL_TRAIN_SEED:--1}"
# 9360min (156h) leaves a ~12h buffer before the 7-day (168h) SLURM hard limit above for the
# post-training checkpoint-verify/comparison-eval/artifact-upload steps, which cost roughly the same
# wall time regardless of how many RL steps ran (eval subset size is fixed).
export SENPAI_TIMEOUT_MINUTES="${SENPAI_TIMEOUT_MINUTES:-9360}"
image="docker://ghcr.io/wandb/senpai@sha256:b72da91555753cf52c4e05fcf53e1d343995d28c5d0a13a1e08f4fb3fdbae5b1"
# Overridable for a non-CoreWeave SLURM cluster whose Pyxis/Enroot mount syntax or shared-storage
# paths differ from this project's own CoreWeave SUNK setup.
container_mounts="${SENPAI_CONTAINER_MOUNTS:-/mnt/data:/mnt/data,/mnt/home:/mnt/home}"

mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" \
  "$BIOREASON_JOB_TMP" "$RL_OUTPUT" "$runtime_root/runtime_logs"
cleanup() {
  rm -rf "$BIOREASON_JOB_TMP" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$BIOREASON_ARTIFACT_ROOT"
  if [[ -n "$source_dir" ]]; then
    rm -rf "$source_dir"
  fi
}
trap cleanup EXIT

unset TMPDIR
srun --container-image="$image" \
  --container-mounts="$container_mounts" \
  --container-workdir="$REPO_DIR" --export=ALL bash -lc '
    set -euo pipefail
    export TMPDIR="$BIOREASON_JOB_TMP"
    nvidia-smi
    uv sync --extra texttrain
    # Reference-data cache (ADR-042): only 2 of the 4 bundle files were ever read from
    # $BIOREASON_REFERENCE_DATA_ROOT, so every job fell through to a live Zenodo download of the
    # whole archive regardless -- a flaky external endpoint (two 504 errors in a row blocked the
    # ADR-041 follow-up sealed-holdout jobs). Now reads/writes all 4 named files, so the cache is
    # complete after the first successful job and every later job (any script) can skip Zenodo entirely.
    REFDATA_FILES=(go-basic.obo IA.txt eval_terms_no_knowledge_2025_03.tsv known_t0.tsv)
    mkdir -p data
    for f in "${REFDATA_FILES[@]}"; do
      [[ -r "$BIOREASON_REFERENCE_DATA_ROOT/data/$f" ]] && cp "$BIOREASON_REFERENCE_DATA_ROOT/data/$f" "data/$f"
    done
    for attempt in 1 2 3; do
      if python scripts/fetch_approved_reference_data.py; then
        break
      fi
      if [[ "$attempt" -eq 3 ]]; then
        echo "approved reference data download failed after 3 attempts" >&2
        exit 1
      fi
      echo "approved reference data download attempt $attempt failed; retrying" >&2
      sleep 5
    done
    mkdir -p "$BIOREASON_REFERENCE_DATA_ROOT/data"
    for f in "${REFDATA_FILES[@]}"; do
      cp -f "data/$f" "$BIOREASON_REFERENCE_DATA_ROOT/data/$f" 2>/dev/null || true
    done
    sft_artifact="$(uv run python scripts/read_artifact_ref.py \
      --receipt "$SFT_ARTIFACT_RECEIPT")"
    uv run --extra texttrain torchrun --standalone --nproc_per_node=8 train.py \
      --stage rl --rl_algo "$RL_ALGO" --train_seed "$RL_TRAIN_SEED" \
      --use_multimodal true \
      --model_name "$RL_MODEL_NAME" \
      --sft_artifact "$sft_artifact" \
      --max_steps "$RL_MAX_STEPS" \
      --num_generations "$RL_NUM_GENERATIONS" \
      --max_completion_length "$RL_MAX_COMPLETION_LENGTH" \
      --proteins_per_step "$RL_PROTEINS_PER_STEP" \
      --rl_advantage_normalization "$RL_ADVANTAGE_NORMALIZATION" \
      --eval_subset_size "$RL_EVAL_SUBSET_SIZE" \
      --esm_layer "$RL_ESM_LAYER" \
      --lora_r "$RL_LORA_R" --lora_alpha "$RL_LORA_ALPHA" --beta "$RL_BETA" \
      --output_dir "$RL_OUTPUT" \
      --wandb_name "rl-grpo-$SLURM_JOB_ID"
    uv run --extra texttrain python scripts/verify_sft_checkpoint.py \
      --checkpoint "$RL_OUTPUT" --max-new-tokens 8
    rl_artifact="$(uv run python scripts/read_artifact_ref.py \
      --receipt "$RL_OUTPUT/wandb_artifact.json")"
    uv run --extra texttrain python scripts/compare_rl_checkpoints.py \
      --sft-artifact "$sft_artifact" \
      --rl-artifact "$rl_artifact" \
      --artifact-root "$BIOREASON_ARTIFACT_ROOT/evaluation" \
      --subset-size "$RL_COMPARISON_SUBSET_SIZE" \
      --split "$RL_COMPARISON_SPLIT" \
      --wandb-name "rl-comparison-$SLURM_JOB_ID" \
      --output "$RL_OUTPUT/comparison.json"
  '
