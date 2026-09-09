"""eval_targets.bioreason_pro_test — the sealed held-out test set `wanglab/bioreason-pro-test-data`.

~8,630 proteins, the paper's temporal holdout (annotations Mar 2023–Feb 2024; training cut off
Nov 2022). The WHOLE split is the sealed test set, so split_mode="all" (no in_split re-hash).

Schema (verified via HF): protein_id, protein_names, protein_function, organism,
subcellular_location, go_ids, go_bp (None when empty), go_mf, go_cc (stringified GO-id lists),
sequence, length, interpro_ids, interpro_location, structure_path, interpro_formatted, go_pred.
NOTE it ships only a `test` split, and has NO `ppi_formatted` column → has_ppi=False (the PPI
context slot is explicitly dropped rather than silently missing; a temporal-leakage-safe absence).
The generated `go_pred` field is also excluded by the approved prompt-field contract.
"""

from __future__ import annotations

from eval_targets.base import EvalTarget

TARGET = EvalTarget(
    name="bioreason_pro_test",
    hf_repo="wanglab/bioreason-pro-test-data",
    hf_config=None,
    hf_split="test",
    split_mode="all",          # the whole `test` split IS the sealed eval set
    gated=False,
    available=True,
    has_ppi=False,             # verified: no ppi_formatted column
    has_interpro=True,         # has interpro_formatted
    gt_columns=("go_mf", "go_bp", "go_cc"),
    context_fields=(),         # fail closed: holdout prompt uses only the protein sequence
)
