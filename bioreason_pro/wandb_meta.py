"""Shared W&B run metadata: the derived tag and the auto-generated note.

Every `wandb.init()` call in this project should pass `tags=derived_tags(...)` and
`notes=build_note(...)` computed from this module, so a run is orientable from the W&B run list the
moment it starts — no cross-referencing plan.md or a Slurm log required, and no separate backfill
pass needed for the baseline layer.

This is one half of the vocabulary in docs/WANDB_TAGS.md. The other half — judgements
(recommended/superseded/null-result/...) and repairs of a condition the config never recorded — still
needs a human to add a table entry in `scripts/apply_run_tags.py`, because nothing in the config can
tell you "this is the run we recommend." What CAN be read from the config is handled here, once, so
it is never duplicated or drifts between the live path and the backfill script.
"""

from __future__ import annotations

import json

# Any of these appearing anywhere in the config means the run touched a licence-incompatible asset.
INCOMPATIBLE_MARKERS = ("esm3", "esm-c", "esmc")

JOB_TYPE_STAGE = {
    "sft-training": "sft",
    "rl-training": "rl",
    "prompt-variant-val-eval": "eval",
    "model-evaluation": "eval",
    "full-holdout-evaluation": "eval",
    "dev-set-evaluation": "eval",
    "agent-iteration": "agent",
    "agent-iteration-finalize": "agent",
}

# Human-readable opener for the note. Falls back to the raw job_type string for anything unlisted,
# so a new call site is never silently unlabelled.
JOB_TYPE_LABEL = {
    "sft-training": "SFT",
    "rl-training": "RL",
    "prompt-variant-val-eval": "val sweep",
    "model-evaluation": "eval",
    "full-holdout-evaluation": "sealed holdout",
    # Distinct from "full-holdout-evaluation": plan.md Phase 3 re-measures existing checkpoints on
    # cafa_no_knowledge (1,496 proteins), which is not the sealed holdout. Sharing one label between
    # the two would put the literal text "sealed holdout" on a run that never touched it — exactly
    # what docs/WANDB_TAGS.md calls "an incident, not a curiosity" for the `sealed` tag itself.
    "dev-set-evaluation": "dev-set eval",
    "agent-iteration": "agent iteration",
    "agent-iteration-finalize": "agent iteration (finalize)",
}

# Config keys worth surfacing in the note, in the order they should appear, each with a short label.
# Deliberately short: the note is a one-line orientation, not a config dump — the config tab already
# holds everything, verbatim, for anyone who needs more than the headline.
_NOTE_FIELDS = (
    ("target_variant", "target"),
    ("reward_variant", "reward"),
    ("prompt_variant", "prompt"),
    ("train_seed", "seed"),
    ("sft_artifact", "sft"),
    ("rl_artifact", "rl"),
    ("artifact", "artifact"),
    ("target", "eval_target"),
    ("split", "split"),
    ("iteration_id", "iteration"),
    ("policy_id", "policy"),
)

_NOTE_MAX_LEN = 300


def provenance(config: dict) -> str:
    """'pre-boundary' if any ESM3/ESM-C marker appears anywhere in the config, else 'approved'."""
    blob = json.dumps(config).lower() if config else ""
    return "pre-boundary" if any(m in blob for m in INCOMPATIBLE_MARKERS) else "approved"


def derived_tags(job_type: str | None, config: dict) -> list[str]:
    """Tags that need no table entry: provenance and stage, computed from config/job_type alone.

    Idempotent and side-effect-free, so it is safe to call both at `wandb.init()` time and again
    from the backfill script without the two ever disagreeing — there is exactly one implementation.
    """
    tags = {provenance(config)}
    stage = JOB_TYPE_STAGE.get(job_type or "")
    if stage:
        tags.add(stage)
    return sorted(tags)


def build_note(job_type: str | None, config: dict, extra: str | None = None) -> str:
    """A one-line, human-readable orientation for the W&B run list.

    Composed only from fields already in `config` (plus one optional free-text `extra` clause for a
    call site that knows something the standard field list doesn't cover), so it carries no
    information the config didn't already have and can never drift out of sync with it.
    """
    label = JOB_TYPE_LABEL.get(job_type or "", job_type or "run")
    parts = [label]
    for key, short in _NOTE_FIELDS:
        value = config.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{short}={value}")
    if extra:
        parts.append(extra)
    note = " · ".join(str(p) for p in parts)
    if len(note) > _NOTE_MAX_LEN:
        note = note[: _NOTE_MAX_LEN - 1].rstrip() + "…"
    return note
