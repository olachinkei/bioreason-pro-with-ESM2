#!/bin/bash
# Distributed immutable SFT-versus-GRPO evaluation on the sealed public holdout.
# Full run (8 GPUs):
#   sbatch --export=ALL,SOURCE_REVISION=<commit>,SOURCE_BUNDLE=<bundle>,SFT_ARTIFACT=<...:vN>,RL_ARTIFACT=<...:vN> slurm/eval_full_holdout.sbatch
# Two-GPU smoke: override GPU/CPU/RAM/time and export
# EVAL_WORLD_SIZE=2,EVAL_SUBSET_SIZE=4.
#SBATCH --job-name=bioreason-eval-full-holdout
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/eval-full-holdout-%j.out
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_REVISION:?submit with the exact 40-character Git commit to run}"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a lowercase 40-character Git commit" >&2
  exit 2
fi
: "${SFT_ARTIFACT:?submit with an immutable SFT Artifact reference}"
: "${RL_ARTIFACT:?submit with an immutable RL Artifact reference}"
if [[ ! "$SFT_ARTIFACT" =~ :v[0-9]+$ || ! "$RL_ARTIFACT" =~ :v[0-9]+$ ]]; then
  echo "SFT_ARTIFACT and RL_ARTIFACT must use immutable :vN versions" >&2
  exit 2
fi
export BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"

runtime_root="${BIOREASON_RUNTIME_ROOT:-/mnt/data/$USER/BioReason-Pro}"
mkdir -p "$runtime_root"
source_dir=""
if [[ -z "${REPO_DIR:-}" ]]; then
  source_dir="$runtime_root/source-eval-full-holdout-$SLURM_JOB_ID"
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
export BIOREASON_ARTIFACT_ROOT="$runtime_root/artifact_inputs/eval-full-holdout-$SLURM_JOB_ID"
export BIOREASON_JOB_TMP="$runtime_root/tmp/eval-full-holdout-$SLURM_JOB_ID"
export EVAL_OUTPUT="${EVAL_OUTPUT:-$runtime_root/artifacts/eval-full-holdout-$SLURM_JOB_ID}"
export EVAL_WORLD_SIZE="${EVAL_WORLD_SIZE:-${SLURM_GPUS_ON_NODE:-8}}"
export EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-}"
export EVAL_MAX_COMPLETION_LENGTH="${EVAL_MAX_COMPLETION_LENGTH:-1024}"
# Defaults reproduce the sealed-holdout behavior byte-for-byte. plan.md Phase 3 overrides both to
# re-measure the same two checkpoints on cafa_no_knowledge instead (a different eval_targets/ target,
# 1,496 proteins rather than 8,630) — compare_full_holdout.py has taken --target/
# --expected-full-size all along, this is just wiring the sbatch layer through to it.
export EVAL_TARGET="${EVAL_TARGET:-bioreason_pro_test}"
export EVAL_EXPECTED_FULL_SIZE="${EVAL_EXPECTED_FULL_SIZE:-8630}"
# Default runs both checkpoints in this one job (unchanged behaviour). Set EVAL_LABELS=sft or
# EVAL_LABELS=sft_to_grpo to submit SFT and RL as two independent jobs (e.g. one node each) that
# run in parallel instead of one job evaluating both checkpoints sequentially -- both artifacts are
# still downloaded and lineage-checked in either case.
export EVAL_LABELS="${EVAL_LABELS:-sft,sft_to_grpo}"
image="docker://ghcr.io/wandb/senpai@sha256:b72da91555753cf52c4e05fcf53e1d343995d28c5d0a13a1e08f4fb3fdbae5b1"
# Overridable for a non-CoreWeave SLURM cluster whose Pyxis/Enroot mount syntax or shared-storage
# paths differ from this project's own CoreWeave SUNK setup.
container_mounts="${SENPAI_CONTAINER_MOUNTS:-/mnt/data:/mnt/data,/mnt/home:/mnt/home}"

mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" \
  "$BIOREASON_JOB_TMP" "$EVAL_OUTPUT" "$runtime_root/runtime_logs"
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
    subset_args=()
    if [[ -n "$EVAL_SUBSET_SIZE" ]]; then
      subset_args=(--subset-size "$EVAL_SUBSET_SIZE")
    fi
    timeout --signal=TERM 23h uv run --extra texttrain torchrun \
      --standalone --nproc_per_node="$EVAL_WORLD_SIZE" \
      scripts/compare_full_holdout.py \
      --sft-artifact "$SFT_ARTIFACT" \
      --rl-artifact "$RL_ARTIFACT" \
      --artifact-root "$BIOREASON_ARTIFACT_ROOT/models" \
      --shard-root "$BIOREASON_JOB_TMP/shards" \
      --target "$EVAL_TARGET" \
      --expected-full-size "$EVAL_EXPECTED_FULL_SIZE" \
      --max-completion-length "$EVAL_MAX_COMPLETION_LENGTH" \
      --wandb-name "eval-full-holdout-$SLURM_JOB_ID" \
      --output "$EVAL_OUTPUT/comparison.json" \
      --labels "$EVAL_LABELS" \
      "${subset_args[@]}"
  '
