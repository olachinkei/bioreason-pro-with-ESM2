#!/bin/bash
# Full approved Phase 3 ESM2 SFT run. Submit from the SUNK login node:
# sbatch --export=ALL,SOURCE_REVISION=<40-character-commit> \
#   slurm/sft.sbatch
# Code, installs, downloads, and training execute only inside this GPU allocation.
#SBATCH --job-name=bioreason-sft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/data/%u/BioReason-Pro/runtime_logs/sft-%j.out
#SBATCH --no-requeue

set -euo pipefail
: "${SOURCE_REVISION:?submit with the exact 40-character Git commit to run}"
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a lowercase 40-character Git commit" >&2
  exit 2
fi
export BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"

export BIOREASON_RUNTIME_ROOT="${BIOREASON_RUNTIME_ROOT:-/mnt/data/$USER/BioReason-Pro}"
mkdir -p "$BIOREASON_RUNTIME_ROOT"
source_dir=""
if [[ -z "${REPO_DIR:-}" ]]; then
  source_dir="$BIOREASON_RUNTIME_ROOT/source-sft-$SLURM_JOB_ID"
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
export BIOREASON_CACHE_ROOT="$BIOREASON_RUNTIME_ROOT/cache"
export HF_HOME="$BIOREASON_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$BIOREASON_CACHE_ROOT/xdg"
# ADR-036/037 incident: wandb's artifact-staging path follows XDG_DATA_HOME (default ~/.local/share,
# on the container's small local root filesystem) independently of XDG_CACHE_HOME/WANDB_DIR — job 1692
# completed training and val, then failed only at artifact upload with "disk quota exceeded" writing
# to /root/.local/share/wandb/artifacts/staging. Redirect it to the same ample shared storage.
export XDG_DATA_HOME="$BIOREASON_CACHE_ROOT/xdg-data"
export TRITON_CACHE_DIR="$BIOREASON_CACHE_ROOT/triton-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="$BIOREASON_CACHE_ROOT/torch_inductor-$SLURM_JOB_ID"
export WANDB_DIR="$BIOREASON_RUNTIME_ROOT/wandb"
export WEAVE_SERVER_CACHE_DIR="$WANDB_DIR/weave_server_cache"
export BIOREASON_JOB_TMP="$BIOREASON_RUNTIME_ROOT/tmp/sft-$SLURM_JOB_ID"
export SFT_OUTPUT="${SFT_OUTPUT:-$BIOREASON_RUNTIME_ROOT/artifacts/sft-$SLURM_JOB_ID}"
# Which ESM2 hidden state the fusion reads. 33 (the default) is the final layer, after
# emb_layer_norm_after. ESM2 is trained on masked-residue prediction, so its last layer is
# specialised for that task while intermediate layers may carry more function-relevant signal —
# the untested hypothesis in issue #95. Changing this changes the embeddings, so it needs its own
# SFT run rather than a decode-time flag.
export SFT_ESM_LAYER="${SFT_ESM_LAYER:-33}"
# Stochastic-training seed only; the data split stays on --seed, so replicate SFTs see identical
# proteins. -1 = follow --seed, which is the shipped behaviour. Without this, the leaf_only
# supervision claim (+0.0715, the largest in the ledger) could not be replicated at all.
export SFT_TRAIN_SEED="${SFT_TRAIN_SEED:--1}"
# ADR-015. false reproduces every SFT in plan.md exactly. true stops supervising the manufactured
# empty <think></think>, which is what made the reasoning axis unreachable from RL.
export SFT_MASK_EMPTY_THINK="${SFT_MASK_EMPTY_THINK:-false}"
# ADR-036/037: unlike every other phase's sbatch script, this one never overrode the shared 256-token
# default, so SFT's own internal val step could never reach a closing </think> tag for the reasoned
# variant's own (measured p95≈1600 tok) reasoning traces — reasoning_nonempty=0 and a near-zero
# weighted_fmax at every SFT seed, a measurement artifact, not a real absence of reasoning. 3072 covers
# the corpus's own measured p99 (≈1900 tok) with headroom for the final-answer GO-term lines.
export SFT_MAX_COMPLETION_LENGTH="${SFT_MAX_COMPLETION_LENGTH:-3072}"

mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$WANDB_DIR" "$WEAVE_SERVER_CACHE_DIR" "$BIOREASON_JOB_TMP" "$SFT_OUTPUT" \
  "$BIOREASON_RUNTIME_ROOT/runtime_logs"
cleanup() {
  rm -rf "$BIOREASON_JOB_TMP" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  if [[ -n "$source_dir" ]]; then
    rm -rf "$source_dir"
  fi
}
trap cleanup EXIT
cd "$REPO_DIR"

hostname
scontrol show job "$SLURM_JOB_ID" | sed -n '1,80p'
SFT_IMAGE="${SFT_IMAGE:-docker://ghcr.io/wandb/senpai@sha256:b72da91555753cf52c4e05fcf53e1d343995d28c5d0a13a1e08f4fb3fdbae5b1}"
# Overridable for a non-CoreWeave SLURM cluster whose Pyxis/Enroot mount syntax or shared-storage
# paths differ from this project's own CoreWeave SUNK setup.
container_mounts="${SENPAI_CONTAINER_MOUNTS:-/mnt/data:/mnt/data,/mnt/home:/mnt/home}"

unset TMPDIR
srun --container-image="$SFT_IMAGE" \
  --container-mounts="$container_mounts" \
  --container-workdir="$REPO_DIR" \
  --export=ALL \
  bash -lc '
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
    uv run --extra texttrain python scripts/verify_esm2_contract.py
    # train.py resolves the default 10 epochs to 5,000 finite optimizer steps.
    uv run --extra texttrain torchrun --standalone --nproc_per_node=8 train.py \
      --stage sft \
      --use_multimodal true \
      --model_name Qwen/Qwen3-4B-Thinking-2507 \
      --esm_model_name facebook/esm2_t33_650M_UR50D \
      --smoke false \
      --epochs 10 \
      --esm_layer "$SFT_ESM_LAYER" \
      --train_seed "$SFT_TRAIN_SEED" \
      --mask_empty_think "$SFT_MASK_EMPTY_THINK" \
      --lora_r 128 \
      --lora_alpha 256 \
      --gradient_accumulation_steps 4 \
      --max_completion_length "$SFT_MAX_COMPLETION_LENGTH" \
      --output_dir "$SFT_OUTPUT" \
      --wandb_name "sft-$SLURM_JOB_ID"
    uv run --extra texttrain python scripts/verify_sft_checkpoint.py \
      --checkpoint "$SFT_OUTPUT"
  '

df -h /mnt/home /mnt/data 2>/dev/null || df -h
du -h -d 2 "$BIOREASON_RUNTIME_ROOT" 2>/dev/null | sort -h | tail -30
