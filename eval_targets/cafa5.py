"""eval_targets.cafa5 — the paper's public eval corpus `wanglab/cafa5` (DECLARED, NOT YET ACTIVE).

This is the ~75 GB gated dataset the paper's own eval.py runs against (config `cafa5_reasoning`,
context toggles --interpro_in_prompt / --ppi_in_prompt / --go_gpt_predictions_column). Running the
It remains disabled until access, license, schema, and column provenance have been reviewed.

It is `available=False` because the HF token has NOT been granted access (dataset_info works,
load_dataset fails "ask for access") AND the exact schema is unverified behind the gate. get_target()
refuses this target until the TODOs below are resolved.

TODO(pending wanglab/cafa5 access):
  1. Request + obtain HF access for the account's token.
  2. Confirm the config name (`cafa5_reasoning`?), the eval split name, and whether an in_split /
     temporal filter is needed (split_mode).
  3. Verify column names: id field, sequence field, ground-truth GO columns, and that
     ppi_formatted / interpro_formatted / go_pred exist (cafa5 carries PPI, unlike test-data).
  4. Set the fields below accordingly and flip available=True; add a schema smoke test.
"""

from __future__ import annotations

from eval_targets.base import EvalTarget

TARGET = EvalTarget(
    name="cafa5",
    hf_repo="wanglab/cafa5",
    hf_config="cafa5_reasoning",   # TODO: confirm once access is granted
    hf_split="test",               # TODO: confirm split name / whether a temporal filter is needed
    split_mode="all",              # TODO: confirm (may need split_mode="hash")
    gated=True,
    available=False,               # flip to True after the TODOs are resolved
    has_ppi=True,                  # cafa5 carries PPI context (higher fidelity than test-data)
    has_interpro=True,
    gt_columns=("go_mf", "go_bp", "go_cc"),  # TODO: verify against the real cafa5 schema
)
