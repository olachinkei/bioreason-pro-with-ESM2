"""eval_targets.cafa_no_knowledge — the temporal development set (plan.md Phase 1, ADR-022/024/029).

Not an HF-hosted dataset: `source="local"`, materialized by `scripts/build_cafa_no_knowledge_dev_set.py`
into `data/cafa_no_knowledge_dev_set.jsonl` (one JSON object per line: protein_id, sequence, go_mf,
go_bp, go_cc, organism, and optionally interpro_formatted).

Why this target exists. ADR-022 found that this repository's *validation* split — a random hash
partition of a 99.7%-pre-annotated training corpus — is a different, easier task than the paper's
temporal holdout: a zero-parameter `interpro2go` lookup beats the trained model there. The corpus
itself cannot supply a temporal dev set (ADR-024: its own val/test hash partitions are ~99.7%
pre-annotated), and the sealed set cannot be used for steering (ADR-002). This target is built
instead from CAFA's own "no-knowledge" evaluation targets — proteins with no prior experimental
annotation in *any* GO aspect as of CAFA t0 — with the 213 ids that overlap the pinned sealed holdout
and the 8 that overlap the training corpus (ADR-029) removed, leaving 1,496 proteins disjoint from
both the sealed test set and the training corpus.

Prompt shape: routed through `bioreason_pro.data_contract.adapt_row_for_evaluation` (see
`eval_targets.base.adapt_row`'s special case for this target's name), NOT this module's generic
context_fields/has_interpro mechanism used by `cafa5`. Phase 3 exists to re-measure the reasoned
checkpoints (`sft-checkpoint:v20`/`rl-checkpoint:v28`) on this dev set, and those were trained on
`data_contract`'s prompt template (system prompt, instruction text, and CONTEXT_COLUMNS rendered with
an explicit "not available" placeholder when missing) — a different template from this module's
generic one. Using the wrong template would confound "corrected split" with "different prompt" in one
number, which is exactly the kind of measurement this project's method rules exist to prevent.

InterPro context: `interpro_formatted` is real once `scripts/fetch_interpro_annotations.py` (EBI
InterProScan, plan.md Phase 1's "blocker, de-risked but not closed") has run for a given protein and
its row in the materialized file carries the field; `ppi_formatted` and `subcellular_location` are
never present for this target (not collected — Phase 1 only tackled the InterPro question), so for a
reasoned-variant evaluation those two always render as "not available", the same stable-absence
handling `bioreason_pro_test` already relies on for its own missing `ppi_formatted`.

Organism (plan.md Phase 5): every row carries a real `organism` value —
`scripts/build_cafa_no_knowledge_dev_set.py` fetches it from the same UniProt REST endpoint the
training corpus's own `organism` column was sourced from (verified byte-identical on shared ids), so
this is the one reasoned-variant prompt section with full coverage rather than a stable placeholder.
"""

from __future__ import annotations

from eval_targets.base import EvalTarget

TARGET = EvalTarget(
    name="cafa_no_knowledge",
    hf_repo="",                # not HF-hosted
    source="local",
    local_path="data/cafa_no_knowledge_dev_set.jsonl",
    split_mode="all",          # the whole materialized file IS the dev set
    gated=False,
    available=True,
    gt_columns=("go_mf", "go_bp", "go_cc"),
    # has_ppi/has_interpro/context_fields are NOT used for this target — adapt_row's special case
    # (module docstring above) determines prompt content from the row itself, not these defaults.
)
