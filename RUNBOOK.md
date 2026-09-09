# RUNBOOK — operating the training/eval pipeline

Operator-facing procedures for launching, monitoring, and recovering SFT, RL, and evaluation jobs on
CoreWeave SUNK.

## 0. Preconditions
- `.env` complete (all keys, incl. `GITHUB_TOKEN` fine-grained + `EXA_API_KEY`). Never commit it.
- `bioreason_pro/approved_assets.json` contains every model used by the selected config.
- Only manifest-listed Apache-2.0/MIT models and CC-BY-4.0 reference data are supported.
- Run `python scripts/fetch_approved_reference_data.py`; GO/IA checksums must pass preflight.
- `data/public_holdout_ids.txt` must match the pinned public dataset fingerprint.
- Precomputed or cached GO embeddings stay disabled until their source-model provenance is approved.
- `wandb/bioreasonpro-senpai` exists with the `schmidhuber` branch + branch protection.
- `uv sync` resolves; `pytest -q` green (pre-GPU).

## 0. Submitting a job

Use the helper rather than composing `sbatch` by hand:

```bash
scripts/submit_slurm.sh slurm/sft.sbatch SENPAI_TARGET_VARIANT=leaf_only
```

It builds the bundle for `HEAD`, copies it to shared storage, proves `git fetch` reaches that exact
revision on the login node, and only then submits. Two failure modes this closes, both of which have
already consumed scheduled jobs:

- submitting after a merge without staging the new revision's bundle — the job starts and dies two
  seconds later on `SOURCE_BUNDLE is not readable`;
- passing a comma-containing value through `sbatch --export=KEY=VALUE`, where commas separate
  assignments, so `PROMPT_COMPLETION_LENGTHS=64,128,256` becomes `=64` plus three bogus entries.

It refuses to submit from a dirty working tree, since the bundle would not contain what you are
testing.

## 0b. Reproducing the best known configuration

The Phase 7 search settled on a specific combination. Every element is opt-in, so running the
launchers with their defaults reproduces the **original** recipe, not this one. All four flags matter:

```bash
# 1. SFT with leaf-only supervision (the largest single lever, +41%)
scripts/submit_slurm.sh slurm/sft.sbatch SENPAI_TARGET_VARIANT=leaf_only

# 2. GRPO from that artifact's receipt, with the aspect-aware reward and 16 rollouts per prompt
scripts/submit_slurm.sh slurm/rl_grpo.sbatch \
  SFT_ARTIFACT_RECEIPT=<runtime_root>/artifacts/sft-<jobid>/wandb_artifact.json \
  SENPAI_TARGET_VARIANT=leaf_only \
  SENPAI_REWARD_VARIANT=aspect_mean \
  RL_NUM_GENERATIONS=16 \
  RL_MAX_STEPS=50 \
  RL_MAX_COMPLETION_LENGTH=64
```

Then evaluate at `max_completion_length=64`, **not** the 1024 default:

```bash
python eval.py --checkpoint <ckpt> --split val --subset_size 256 --diagnostics
```

Validation weighted F_max ~`0.310` against ~`0.112` for the shipped recipe at its shipped budget.
The generation budget must be swept per arm — a comparison at one shared budget measures the budget
rather than the model.

Two knobs that look helpful and are not: `RL_MAX_STEPS=100` costs ~10%, and
`SFT_ESM_LAYER=24` gives a better SFT but a *worse* post-RL model.

## 1. Preflight

The login host is `sunk.cwb607-trainingsss.coreweave.app` — **note the `sss`**. A host without it
(`sunk.cwb607-training.coreweave.app`) resolves to a different cluster with no nodes registered at
all, where `sinfo` reports 0 nodes and every job sits in `PENDING (PartitionConfig)` indefinitely.
Confirm the login banner says `trainingsss-login-...` before submitting anything.

```bash
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-trainingsss.coreweave.app
```

Confirm Slurm, container runtime, shared storage, GPU partition/QOS/RAM, `.env` completeness, and
the pinned image digest before submitting anything.

### SFT: GPU smoke and full run

Use the one-GPU launcher for the smoke. It publishes and finalizes the SFT model artifact; do not
run model imports on the login node:

First, create a Git bundle from the exact revision on the workstation and place it under
`/mnt/data/$USER/BioReason-Pro/source` on the cluster. This lets compute nodes load private source
without receiving GitHub credentials:

```bash
SOURCE_REVISION="$(git rev-parse origin/main)"
SOURCE_BUNDLE_LOCAL="/tmp/bioreasonpro-$SOURCE_REVISION.bundle"
git bundle create "$SOURCE_BUNDLE_LOCAL" origin/main
ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-trainingsss.coreweave.app \
  'mkdir -p "/mnt/data/$USER/BioReason-Pro/source"'
scp -o IdentitiesOnly=yes "$SOURCE_BUNDLE_LOCAL" \
  "kkamata+cwb607@sunk.cwb607-trainingsss.coreweave.app:/mnt/data/kkamata+cwb607/BioReason-Pro/source/"
```

Then, on the login node:

```bash
SOURCE_REVISION=<40-character-commit>
SOURCE_BUNDLE="/mnt/data/$USER/BioReason-Pro/source/bioreasonpro-$SOURCE_REVISION.bundle"
mkdir -p "/mnt/data/$USER/BioReason-Pro/runtime_logs"
sbatch --export=ALL,SOURCE_REVISION="$SOURCE_REVISION",SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  slurm/sft_smoke.sbatch
```

After the smoke and reload verification pass, submit the full approved-data run from the login
node:

```bash
sbatch --export=ALL,SOURCE_REVISION="$SOURCE_REVISION",SOURCE_BUNDLE="$SOURCE_BUNDLE" \
  slurm/sft.sbatch
```

The full launcher redirects Hugging Face, W&B, Weave, compiler, temporary, and checkpoint state to
`/mnt/data/$USER/BioReason-Pro`, uses eight H100s, logs loss/gradient norm/throughput/GPU memory and
validation metrics to W&B, saves a complete checkpoint, then reload-verifies it.
The training run must also finalize its model artifact and write the immutable reference to
`$SFT_OUTPUT/wandb_artifact.json`; a local checkpoint alone does not unblock RL.

### RL (GRPO): bounded run

Queue the RL job behind the successful SFT job so RL can only read the finalized SFT artifact
receipt:

```bash
sbatch --dependency=afterok:<sft-job-id> \
  --export=ALL,SOURCE_REVISION="$SOURCE_REVISION",SOURCE_BUNDLE="$SOURCE_BUNDLE",SFT_ARTIFACT_RECEIPT=/mnt/data/$USER/BioReason-Pro/artifacts/sft-smoke-<sft-job-id>/wandb_artifact.json \
  slurm/rl_grpo_smoke.sbatch
```

All launchers require the exact 40-character commit and check out that revision in detached mode
inside the allocation. `SOURCE_BUNDLE` is preferred for private source; an authenticated
`SOURCE_REPOSITORY_URL` remains available as a fallback. A movable or deleted branch name is not
accepted as an experiment input.

The active RL run calls `use_artifact` with the receipt's immutable version before model loading,
then publishes and finalizes its own model artifact. The expected graph is
`SFT artifact → RL run → RL artifact`. The comparison run independently calls `use_artifact` for
both versions before writing `comparison.json` with identical greedy decoding, completion limit,
subset, public target, and IA-weighted scoring.

After the smoke succeeds, submit `slurm/rl_grpo.sbatch` with the same source and receipt
contract. The reference defaults to 100 GRPO optimizer steps on eight H100s and evaluates both
immutable checkpoints on the same 64-example public-holdout slice. Override the bounded values with
`RL_MAX_STEPS`, `RL_NUM_GENERATIONS`, `RL_MAX_COMPLETION_LENGTH`,
`RL_EVAL_SUBSET_SIZE`, or `RL_COMPARISON_SUBSET_SIZE` when a reviewed experiment requires
different limits.

The launchers first seed `data/go-basic.obo` and `data/IA.txt` from the optional shared
`$BIOREASON_REFERENCE_DATA_ROOT/data` cache, then run the repository checksum validator before use.
If the cache is absent, they download the pinned CC-BY-4.0 archive with three bounded attempts.

### Eval: full public holdout

Use only the immutable model versions finalized by the SFT and RL jobs. First validate the distributed
shard and aggregation path with two GPUs and four proteins:

```bash
sbatch --gres=gpu:h100:2 --cpus-per-task=24 --mem=450G --time=02:00:00 \
  --export=ALL,SOURCE_REVISION="$SOURCE_REVISION",SOURCE_BUNDLE="$SOURCE_BUNDLE",EVAL_WORLD_SIZE=2,EVAL_SUBSET_SIZE=4,SFT_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:v2,RL_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:v2 \
  slurm/eval_full_holdout.sbatch
```

When the smoke has finalized its W&B evaluation Artifact, run all 8,630 sealed proteins by omitting
the two smoke overrides:

```bash
sbatch \
  --export=ALL,SOURCE_REVISION="$SOURCE_REVISION",SOURCE_BUNDLE="$SOURCE_BUNDLE",SFT_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-sft-checkpoint:v2,RL_ARTIFACT=wandb-healthcare/bioreasonpro-senpai/bioreasonpro-rl-checkpoint:v2 \
  slurm/eval_full_holdout.sbatch
```

Each GPU evaluates a deterministic disjoint shard for both models. Rank 0 refuses incomplete,
duplicate, or mismatched model sets before scoring, logs both immutable Artifact inputs to the same
W&B run, and publishes `bioreasonpro-phase5-full-holdout:vN`. Job-specific model downloads, source,
shards, compiler caches, and temporary files are removed on exit; the result, Artifact receipt, and
runtime log are retained. Do not use the full-holdout result for hyperparameter tuning.

## 2. Monitor
```bash
squeue -u "$USER"                                            # jobs
tail -f /mnt/data/$USER/BioReason-Pro/runtime_logs/*.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
# W&B: runs under wandb-healthcare/bioreasonpro-senpai ; Weave: filter rollouts by
#   inputs.group_id / inputs.rollout_index / attrs stage,rl_algo,rl_global_step
```
Watch for: PENDING-too-long (RAM/partition) and NaN/flat loss.

## 3. Recover
A job that dies partway through (transient network error, node failure) can simply be resubmitted
with the same `scripts/submit_slurm.sh` command — every launcher is idempotent given the same
inputs, and RL resumes are keyed off the SFT artifact receipt, not local state.
