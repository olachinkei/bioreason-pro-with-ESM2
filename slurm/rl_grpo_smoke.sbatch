#!/bin/bash
# Submit after a successful SFT smoke:
# sbatch --dependency=afterok:<sft-smoke-job> \
#   --export=ALL,SOURCE_REVISION=<40-character-commit>,SFT_ARTIFACT_RECEIPT=/mnt/data/$USER/BioReason-Pro/artifacts/sft-smoke-<job>/wandb_artifact.json \
#   slurm/rl_grpo_smoke.sbatch
#SBATCH --job-name=bioreason-rl-grpo-smoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/rl-grpo-smoke-%j.out
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
  source_dir="$runtime_root/source-rl-grpo-smoke-$SLURM_JOB_ID"
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
export BIOREASON_ARTIFACT_ROOT="$runtime_root/artifact_inputs/rl-grpo-smoke-$SLURM_JOB_ID"
export BIOREASON_JOB_TMP="$runtime_root/tmp/rl-grpo-smoke-$SLURM_JOB_ID"
export RL_OUTPUT="${RL_OUTPUT:-$runtime_root/artifacts/rl-grpo-smoke-$SLURM_JOB_ID}"
export RL_MODEL_NAME="${RL_MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}"
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
    uv run --extra texttrain python train.py \
      --stage rl --rl_algo grpo --smoke true --use_multimodal true \
      --model_name "$RL_MODEL_NAME" \
      --sft_artifact "$sft_artifact" \
      --max_steps 2 --num_generations 2 --max_completion_length 8 \
      --proteins_per_step 2 --rl_advantage_normalization group_mean_global_std \
      --eval_subset_size 1 --lora_r 16 --lora_alpha 32 --beta 0.0 \
      --output_dir "$RL_OUTPUT" \
      --wandb_name "rl-grpo-smoke-$SLURM_JOB_ID"
    uv run --extra texttrain python scripts/verify_sft_checkpoint.py \
      --checkpoint "$RL_OUTPUT" --max-new-tokens 8
    rl_artifact="$(uv run python scripts/read_artifact_ref.py \
      --receipt "$RL_OUTPUT/wandb_artifact.json")"
    uv run --extra texttrain python scripts/compare_rl_checkpoints.py \
      --sft-artifact "$sft_artifact" \
      --rl-artifact "$rl_artifact" \
      --artifact-root "$BIOREASON_ARTIFACT_ROOT/evaluation-$SLURM_JOB_ID" \
      --subset-size 1 \
      --wandb-name "rl-comparison-smoke-$SLURM_JOB_ID" \
      --output "$RL_OUTPUT/comparison.json"
  '
