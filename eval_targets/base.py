"""eval_targets.base — the held-out-dataset abstraction shared by every eval target.

A sealed-test evaluation needs to take a raw HF row from some held-out dataset, assemble the exact
prompt the fusion model was trained on, and expose the ground-truth GO set — WITHOUT touching the
PROTECTED `data.py`/`eval.py`. This module is that seam:

  * `EvalTarget`         — a frozen spec: which HF repo/split, what context the schema carries, how
                           to select the eval subset (whole split vs. protein-hash filter).
  * `adapt_row`          — raw row -> the pre-assembled `prompt` dict `data.format_cafa5_for_protein_llm`
                           expects (the raw wanglab schema has no `prompt` key). Generalized from
                           train.py::_adapt_real_row; the prompt text is kept byte-identical to
                           training so eval fidelity holds.
  * `parse_gt`           — raw row -> ground-truth GO set (unions the aspect columns; tolerates None
                           and stringified lists). Mirrors train.py::_parse_go.
  * `stream_eval_records`— yields the `{protein_id, user_turn, sequence, gt_terms}` records that
                           train.run_sealed_eval feeds to eval.score_generations.

Depends only on the PROTECTED `data.py` pure helpers (imported, never edited) + `datasets` + stdlib.
It deliberately does NOT import `train` (train depends on this package, not the reverse).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Kept identical to train.py::_SYS_PROMPT / _adapt_real_row so the sealed-eval prompt matches the
# prompt the model saw during SFT/RL (any drift here would silently change the measured F_max).
SYS_PROMPT = "You are an expert protein-function-prediction assistant."
USER_INSTRUCTION = (
    "Predict the Gene Ontology (GO) terms for this protein and give a functional summary. "
    "Reason step by step, then give the final answer.\n\n"
)

# Optional decode-time prompt variants, selected by SENPAI_EVAL_PROMPT_VARIANT. The default is
# "baseline", which reproduces USER_INSTRUCTION byte-for-byte — any other value is an explicit
# experiment and is recorded alongside the metric, because prompt drift silently moves F_max.
#
# "specific_only" exists because scoring propagates predictions: naming an ancestor adds nothing
# the scorer would not add for free, and on the Phase 4 rollouts 65.2% of every completion was
# spent that way (see scripts/diagnose_ancestor_redundancy.py). The variant asks the model to
# spend its fixed generation budget on leaves instead.
PROMPT_VARIANTS = {
    "baseline": "",
    "specific_only": (
        "List only the most specific GO terms that apply. Do not list a term when you are also "
        "listing one of its descendants: ancestor terms are added automatically when your answer "
        "is scored, so listing them wastes space you could spend on more specific terms.\n\n"
    ),
}
DEFAULT_PROMPT_VARIANT = "baseline"


def active_prompt_variant() -> str:
    """Read the requested variant, failing closed on an unknown name."""
    import os

    name = os.environ.get("SENPAI_EVAL_PROMPT_VARIANT", DEFAULT_PROMPT_VARIANT).strip()
    if name not in PROMPT_VARIANTS:
        raise ValueError(
            f"SENPAI_EVAL_PROMPT_VARIANT={name!r} is not a known prompt variant; "
            f"expected one of {sorted(PROMPT_VARIANTS)}"
        )
    return name


def prompt_variant_suffix(variant: str | None = None) -> str:
    """Just the variant's extra guidance — empty string for the baseline.

    The sealed-test path assembles its own instruction (`user_instruction`), while the in-loop val
    path reuses the prompt text that ships with the dataset row. Both need the same suffix, so it
    lives here once rather than being written out twice.
    """
    name = variant if variant is not None else active_prompt_variant()
    if name not in PROMPT_VARIANTS:
        raise ValueError(
            f"unknown prompt variant {name!r}; expected one of {sorted(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[name]


def user_instruction(variant: str | None = None) -> str:
    """USER_INSTRUCTION plus the selected variant's extra guidance (empty for the baseline)."""
    return USER_INSTRUCTION + prompt_variant_suffix(variant)
# (label, row-key) context slots, in the paper's order. `interpro_formatted`/`ppi_formatted` are
# gated by the target's has_interpro/has_ppi flags; the rest are included when present.
DEFAULT_CONTEXT_FIELDS = (
    ("Organism", "organism"),
    ("Protein names", "protein_names"),
    ("Subcellular location", "subcellular_location"),
    ("InterPro domains", "interpro_formatted"),
    ("Interaction partners", "ppi_formatted"),
)

_GO_RE = re.compile(r"GO:\d{7}")


@dataclass(frozen=True)
class EvalTarget:
    """Immutable spec for one held-out evaluation dataset."""

    name: str
    hf_repo: str
    hf_config: str | None = None          # HF `datasets` config name (None = default config)
    hf_split: str = "test"                # the HF split to stream (test-data ships only a `test` split)
    split_mode: str = "all"               # "all" = the whole hf_split IS the eval set (sealed test);
                                          # "hash" = additionally filter by data.in_split(pid, ...)
    logical_split: str = "test"           # used only when split_mode == "hash"
    gated: bool = False                   # HF access-gated repo
    available: bool = True                # False -> get_target() refuses (access/schema not confirmed)
    has_ppi: bool = True                  # include the `ppi_formatted` context slot
    has_interpro: bool = True             # include the `interpro_formatted` context slot
    gt_columns: tuple[str, ...] = ("go_mf", "go_bp", "go_cc")
    context_fields: tuple[tuple[str, str], ...] = DEFAULT_CONTEXT_FIELDS
    seq_field: str = "sequence"
    id_field: str = "protein_id"
    source: str = "hf"                    # "hf" (default) or "local" (materialized file, no HF repo)
    local_path: str | None = None         # required when source == "local"; a local JSONL file


def _parse_go(v) -> set[str]:
    """GO-term set from a list, a stringified list, or GO:ids in free text; None/other -> empty set.

    Mirrors train.py::_parse_go so ground-truth parsing matches the training/reward path exactly.
    """
    if isinstance(v, (list, tuple, set)):
        return {t for t in v if isinstance(t, str) and t.startswith("GO:")}
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
        except Exception:
            return set(_GO_RE.findall(v))
        return _parse_go(parsed) if isinstance(parsed, (list, tuple, set)) else set()
    return set()


def parse_gt(row: dict, target: EvalTarget) -> set[str]:
    """Union the target's ground-truth GO columns into one set (aspect-agnostic; tolerates None)."""
    gt: set[str] = set()
    for col in target.gt_columns:
        gt |= _parse_go(row.get(col))
    return gt


def _skip_context_field(key: str, target: EvalTarget) -> bool:
    """Honor the schema flags: drop PPI/InterPro slots the target declares absent (explicit, not silent)."""
    if key == "ppi_formatted" and not target.has_ppi:
        return True
    if key == "interpro_formatted" and not target.has_interpro:
        return True
    return False


def adapt_row(row: dict, target: EvalTarget) -> dict:
    """Raw HF row -> the example shape `data.format_cafa5_for_protein_llm` consumes.

    Assembles the user prompt from the target's context slots (generalized from
    train.py::_adapt_real_row). assistant_reasoning/answer are empty for eval (held-out rows carry no
    reasoning/final_answer, and only the user turn + generated response matter). The sequence is capped
    at data.MAX_LENGTH_PROTEIN to keep the fusion shape contract (ESM2's declared 1,026 positions
    minus BOS/EOS gives 1,024 residues).
    """
    import data

    if target.name == "bioreason_pro_test":
        from bioreason_pro.data_contract import adapt_approved_training_row

        approved = adapt_approved_training_row(
            row, repo_id=target.hf_repo, use="holdout-evaluation",
            max_length_protein=data.MAX_LENGTH_PROTEIN,
        )
        approved["prompt"]["assistant_reasoning"] = ""
        approved["prompt"]["assistant_answer"] = ""
        return approved

    if target.name == "cafa_no_knowledge":
        # Must reproduce the exact prompt shape sft-checkpoint:v20/rl-checkpoint:v28 were trained on
        # (bioreason_pro.data_contract._adapt_fields — a DIFFERENT template from this module's
        # generic path below), or Phase 3's re-measurement confounds "corrected split" with
        # "different prompt" in one number. No `require_approved_dataset` check here: this target's
        # approval is `require_approved_local_eval_target`, already enforced in `_open_target_stream`.
        from bioreason_pro.data_contract import adapt_row_for_evaluation

        approved = adapt_row_for_evaluation(row, source_dataset=target.name,
                                            max_length_protein=data.MAX_LENGTH_PROTEIN)
        approved["prompt"]["assistant_reasoning"] = ""
        approved["prompt"]["assistant_answer"] = ""
        return approved

    ctx: list[str] = []
    for label, key in target.context_fields:
        if _skip_context_field(key, target):
            continue
        v = row.get(key)
        if v:
            block = f"{label}:\n{v}" if ("\n" in str(v) or key.endswith("formatted")) else f"{label}: {v}"
            ctx.append(block)
    user = user_instruction() + "\n\n".join(ctx)
    seq = (row.get(target.seq_field) or "")[: data.MAX_LENGTH_PROTEIN]
    return {
        "prompt": {"system": SYS_PROMPT, "user": user,
                   "assistant_reasoning": row.get("reasoning", "") or "",
                   "assistant_answer": row.get("final_answer", "") or ""},
        "sequence": seq,
        "protein_id": row.get(target.id_field, ""),
        # carried through so downstream code that reads go_* off the example still works
        "go_mf": row.get("go_mf", []), "go_bp": row.get("go_bp", []), "go_cc": row.get("go_cc", []),
    }


def _local_jsonl_rows(path: str):
    """Yield dict rows from a local JSONL eval-target file (line-delimited JSON objects)."""
    import json

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _open_target_stream(target: EvalTarget):
    """Resolve the row source for a target, enforcing the target's own provenance/licence gate.

    `source="hf"` (the sealed HF-hosted targets) requires a pinned, approved dataset revision.
    `source="local"` (e.g. `cafa_no_knowledge`, plan.md Phase 1) is materialized by a repository
    script from externally-fetched, individually-provenanced sources rather than one pinned HF
    revision, so it is gated by `require_approved_local_eval_target` instead.
    """
    if target.source == "local":
        from bioreason_pro.license_policy import require_approved_local_eval_target

        require_approved_local_eval_target(target.name, "dev-evaluation")
        if not target.local_path:
            raise ValueError(f"eval target {target.name!r} has source='local' but no local_path")
        return _local_jsonl_rows(target.local_path)

    from datasets import load_dataset
    from bioreason_pro.license_policy import require_approved_dataset

    revision = require_approved_dataset(target.hf_repo, "holdout-evaluation")
    return load_dataset(
        target.hf_repo,
        target.hf_config,
        split=target.hf_split,
        streaming=True,
        revision=revision,
    )


def stream_eval_records(
    target: EvalTarget,
    seed: int = 0,
    limit: int | None = None,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
):
    """Stream `{protein_id, user_turn, sequence, gt_terms}` records for the target.

    split_mode="all"  -> the whole HF split is the sealed eval set (NO in_split filter — that would
                         re-hash the IDs and keep only ~10%, which is wrong for the sealed test).
    split_mode="hash" -> additionally filter to `logical_split` via the protected data.in_split.

    Streams with an explicit `hf_split` (NOT data._load_stream, which hardcodes split="train" — the
    test-data repo ships only a `test` split). `limit` caps the global row count before sharding
    (smoke runs / subset evals). Shards are a deterministic, disjoint round-robin partition of that
    same prefix, so joining every shard reconstructs exactly the single-process evaluation set.
    """
    import data

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    stream = _open_target_stream(target)
    eligible_index = 0
    for row in stream:
        pid = row.get(target.id_field)
        seq = row.get(target.seq_field)
        if not pid or not seq:
            continue
        if target.split_mode == "hash" and not data.in_split(pid, target.logical_split, seed):
            continue
        if limit is not None and eligible_index >= limit:
            break
        take_row = eligible_index % num_shards == shard_index
        eligible_index += 1
        if not take_row:
            continue
        example = adapt_row(row, target)
        formatted = data.format_cafa5_for_protein_llm(example)
        yield {
            "protein_id": pid,
            "user_turn": formatted["prompt"][0],
            "sequence": formatted["protein_sequences"][0],
            "gt_terms": parse_gt(row, target),
        }
