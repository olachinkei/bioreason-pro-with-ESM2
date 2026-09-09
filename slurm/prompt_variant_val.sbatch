#!/bin/bash
# Learning-free prompt-variant A/B on the val steering split (one H100).
#   sbatch --export=ALL,SOURCE_REVISION=<commit>,SOURCE_BUNDLE=<bundle>,MODEL_ARTIFACT=<...:vN> \
#     slurm/prompt_variant_val.sbatch
# Decodes the same weights twice — baseline vs specific_only — on the same deterministic val
# subset. The sealed holdout is never read.
#SBATCH --job-name=bioreason-prompt-variant-val
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/prompt-variant-val-%j.out
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_REVISION:?submit with the exact 40-character Git commit to run}"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a lowercase 40-character Git commit" >&2
  exit 2
fi
: "${MODEL_ARTIFACT:?submit with an immutable model Artifact reference}"
if [[ ! "$MODEL_ARTIFACT" =~ :v[0-9]+$ ]]; then
  echo "MODEL_ARTIFACT must use an immutable :vN version" >&2
  exit 2
fi
export BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"

runtime_root="${BIOREASON_RUNTIME_ROOT:-/mnt/data/$USER/BioReason-Pro}"
mkdir -p "$runtime_root"
source_dir=""
if [[ -z "${REPO_DIR:-}" ]]; then
  source_dir="$runtime_root/source-promptvariant-$SLURM_JOB_ID"
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
export BIOREASON_ARTIFACT_ROOT="$runtime_root/artifact_inputs/promptvariant-$SLURM_JOB_ID"
export BIOREASON_JOB_TMP="$runtime_root/tmp/promptvariant-$SLURM_JOB_ID"
export PROMPT_VARIANT_OUTPUT="${PROMPT_VARIANT_OUTPUT:-$runtime_root/artifacts/prompt-variant-$SLURM_JOB_ID}"
export PROMPT_VARIANT_SUBSET_SIZE="${PROMPT_VARIANT_SUBSET_SIZE:-256}"
export PROMPT_VARIANT_MAX_COMPLETION_LENGTH="${PROMPT_VARIANT_MAX_COMPLETION_LENGTH:-1024}"
export PROMPT_VARIANTS="${PROMPT_VARIANTS:-baseline,specific_only}"
export PROMPT_COMPLETION_LENGTHS="${PROMPT_COMPLETION_LENGTHS:-}"
image="docker://ghcr.io/wandb/senpai@sha256:b72da91555753cf52c4e05fcf53e1d343995d28c5d0a13a1e08f4fb3fdbae5b1"
# Overridable for a non-CoreWeave SLURM cluster whose Pyxis/Enroot mount syntax or shared-storage
# paths differ from this project's own CoreWeave SUNK setup.
container_mounts="${SENPAI_CONTAINER_MOUNTS:-/mnt/data:/mnt/data,/mnt/home:/mnt/home}"

mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" \
  "$BIOREASON_JOB_TMP" "$PROMPT_VARIANT_OUTPUT" "$runtime_root/runtime_logs"
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
    sweep_args=()
    if [[ -n "$PROMPT_COMPLETION_LENGTHS" ]]; then
      sweep_args=(--completion-lengths "$PROMPT_COMPLETION_LENGTHS")
    fi
    timeout --signal=TERM 210m uv run --extra texttrain python \
      scripts/compare_prompt_variants.py \
      --artifact "$MODEL_ARTIFACT" \
      --artifact-root "$BIOREASON_ARTIFACT_ROOT/model" \
      --subset-size "$PROMPT_VARIANT_SUBSET_SIZE" \
      --max-completion-length "$PROMPT_VARIANT_MAX_COMPLETION_LENGTH" \
      --variants "$PROMPT_VARIANTS" \
      --wandb-name "prompt-variant-val-$SLURM_JOB_ID" \
      --output "$PROMPT_VARIANT_OUTPUT/comparison.json" \
      "${sweep_args[@]}"
  '
