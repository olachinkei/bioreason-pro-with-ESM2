#!/bin/bash
# GO-GPT baseline (plan.md Phase 5 / ADR-028): score the paper's own zero-LLM GO-term predictor on
# the Phase 1 cafa_no_knowledge dev set. Single GPU, inference only, no training.
#
# Smoke (a handful of proteins, ~minutes):
#   scripts/submit_slurm.sh slurm/eval_gogpt_baseline.sbatch GOGPT_LIMIT=3
# Full run (1,496 proteins):
#   scripts/submit_slurm.sh slurm/eval_gogpt_baseline.sbatch
#SBATCH --job-name=bioreason-gogpt-baseline
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/gogpt-baseline-%j.out
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_REVISION:?submit with the exact 40-character Git commit to run}"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a lowercase 40-character Git commit" >&2
  exit 2
fi
export BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"

runtime_root="${BIOREASON_RUNTIME_ROOT:-/mnt/data/$USER/BioReason-Pro}"
mkdir -p "$runtime_root"
source_dir=""
if [[ -z "${REPO_DIR:-}" ]]; then
  source_dir="$runtime_root/source-gogpt-$SLURM_JOB_ID"
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
export XDG_CACHE_HOME="$runtime_root/cache/xdg"
# ADR-036/037 incident: wandb's artifact-staging path follows XDG_DATA_HOME (default ~/.local/share,
# on the container's small local root filesystem) independently of XDG_CACHE_HOME/WANDB_DIR -- job 1692
# completed training and val, then failed only at artifact upload with "disk quota exceeded" writing
# to /root/.local/share/wandb/artifacts/staging. Redirect it to the same ample shared storage.
export XDG_DATA_HOME="$runtime_root/cache/xdg-data"
export BIOREASON_JOB_TMP="$runtime_root/tmp/gogpt-baseline-$SLURM_JOB_ID"
export GOGPT_OUTPUT="${GOGPT_OUTPUT:-$runtime_root/artifacts/gogpt-baseline-$SLURM_JOB_ID}"
export GOGPT_LIMIT="${GOGPT_LIMIT:-}"
image="docker://ghcr.io/wandb/senpai@sha256:b72da91555753cf52c4e05fcf53e1d343995d28c5d0a13a1e08f4fb3fdbae5b1"
# Overridable for a non-CoreWeave SLURM cluster whose Pyxis/Enroot mount syntax or shared-storage
# paths differ from this project's own CoreWeave SUNK setup.
container_mounts="${SENPAI_CONTAINER_MOUNTS:-/mnt/data:/mnt/data,/mnt/home:/mnt/home}"

mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$BIOREASON_JOB_TMP" "$GOGPT_OUTPUT" "$runtime_root/runtime_logs"
cleanup() {
  rm -rf "$BIOREASON_JOB_TMP"
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
    # gogpt/src/gogpt/__init__.py eagerly imports its (unused here) training wrapper, which needs
    # pytorch-lightning -- not part of texttrain (deliberately kept light, no vllm/unsloth
    # co-resolution). This one package is the whole cost of avoiding the much heavier `train` extra.
    # `uv pip install` (not `uv run ... pip install`) so it lands directly in the synced venv without
    # going through another project-sync pass -- job 1296 showed `uv run --extra texttrain pip
    # install X` followed by a later separate `uv run --extra texttrain` re-syncing X back out again.
    uv pip install "pytorch-lightning>=2.3"
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
    limit_args=()
    if [[ -n "$GOGPT_LIMIT" ]]; then
      limit_args=(--limit "$GOGPT_LIMIT")
    fi
    # --no-sync: a plain `uv run --extra texttrain` here would re-sync the venv to exactly
    # pyproject.tomls declared deps and drop the pytorch-lightning just installed above (job 1296).
    uv run --extra texttrain --no-sync python scripts/run_gogpt_baseline.py \
      --out "$GOGPT_OUTPUT" \
      "${limit_args[@]}"
  '
