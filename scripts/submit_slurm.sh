#!/usr/bin/env bash
# Submit a Slurm job at the current revision, staging its source bundle first.
#
# The phase sbatch scripts fetch their source from a git bundle on shared storage. Nothing checks
# that the bundle for the revision you are submitting actually exists, so submitting straight after
# a merge fails two seconds in with "SOURCE_BUNDLE is not readable" — after the job has already been
# scheduled and its slot consumed. This builds the bundle, copies it, proves `git fetch` works on the
# login node, and only then submits.
#
# Usage:
#   SENPAI_CLUSTER_HOST=user@login.example.com SENPAI_REMOTE_ROOT=/mnt/data/user/BioReason-Pro \
#     scripts/submit_slurm.sh slurm/sft.sbatch SENPAI_TARGET_VARIANT=leaf_only
#   scripts/submit_slurm.sh slurm/prompt_variant_val.sbatch \
#       MODEL_ARTIFACT=entity/project/name:v4 PROMPT_COMPLETION_LENGTHS=32,64
#
# Values may contain commas: they are exported in the submitting shell and carried by
# `--export=ALL`, never passed through `sbatch --export=K=V` where commas separate assignments.
#
# SENPAI_CLUSTER_HOST and SENPAI_REMOTE_ROOT are required and have no default: every sbatch script
# in this repo was written for one specific CoreWeave SUNK account, and silently reusing that
# account's login host/shared-storage path on a different cluster fails in a way that looks like a
# transient network issue rather than a config problem. Point these at your own login host and a
# writable directory on your own cluster's shared filesystem.
#
# SENPAI_SBATCH_ARGS optionally overrides sbatch directives baked into the launcher files (e.g.
# `--partition=h100 --gres=gpu:h100:8`) that assume this project's own CoreWeave partition/GPU-type
# names -- command-line sbatch flags take precedence over a script's own `#SBATCH` lines, so this
# works without editing the launcher itself: SENPAI_SBATCH_ARGS="--partition=a100 --gres=gpu:a100:8".
set -euo pipefail

: "${SENPAI_CLUSTER_HOST:?set SENPAI_CLUSTER_HOST to your own cluster's login host, e.g. user@login.example.com}"
: "${SENPAI_REMOTE_ROOT:?set SENPAI_REMOTE_ROOT to a writable directory on your own cluster's shared filesystem}"
HOST="$SENPAI_CLUSTER_HOST"
REMOTE_ROOT="$SENPAI_REMOTE_ROOT"
SBATCH_ARGS="${SENPAI_SBATCH_ARGS:-}"
SSH_OPTS=(-o IdentitiesOnly=yes -o BatchMode=yes)

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <sbatch-path> [KEY=VALUE ...]" >&2
  exit 2
fi
SBATCH_PATH="$1"
shift

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to submit: the working tree is dirty, so the bundle would not match HEAD" >&2
  echo "commit or stash first — the cluster runs the bundled revision, not your working tree." >&2
  exit 2
fi
REV="$(git rev-parse HEAD)"
BUNDLE_LOCAL="$(mktemp -d)/source-$REV.bundle"

echo "==> building bundle for $REV"
git bundle create "$BUNDLE_LOCAL" main >/dev/null
echo "==> copying to $HOST"
scp "${SSH_OPTS[@]}" -q "$BUNDLE_LOCAL" "$HOST:$REMOTE_ROOT/"

echo "==> verifying the bundle fetches on the login node"
ssh "${SSH_OPTS[@]}" "$HOST" "set -e
  B=$REMOTE_ROOT/source-$REV.bundle
  [ -r \"\$B\" ] || { echo 'bundle not readable after copy' >&2; exit 1; }
  T=\$(mktemp -d); git init -q \"\$T\"
  git -C \"\$T\" fetch -q \"\$B\" $REV
  git -C \"\$T\" checkout -q --detach FETCH_HEAD
  [ \"\$(git -C \"\$T\" rev-parse HEAD)\" = '$REV' ] || { echo 'bundle revision mismatch' >&2; exit 1; }
  rm -rf \"\$T\""

echo "==> submitting $SBATCH_PATH"
EXPORTS=""
for kv in "$@"; do
  [[ "$kv" == *=* ]] || { echo "arguments must be KEY=VALUE, got: $kv" >&2; exit 2; }
  EXPORTS+="export ${kv%%=*}='${kv#*=}'"$'\n'
done

ssh "${SSH_OPTS[@]}" "$HOST" "set -e
  REV=$REV; ROOT=$REMOTE_ROOT; B=\$ROOT/source-\$REV.bundle
  CO=\$ROOT/submit-\$REV
  rm -rf \"\$CO\"; git init -q \"\$CO\"
  git -C \"\$CO\" fetch -q \"\$B\" \"\$REV\"
  git -C \"\$CO\" checkout -q --detach FETCH_HEAD
  cd \"\$CO\"
  export SOURCE_REVISION=\$REV SOURCE_BUNDLE=\$B
  $EXPORTS
  sbatch $SBATCH_ARGS --export=ALL $SBATCH_PATH"
