#!/usr/bin/env python3
"""Full-corpus completeness audit (plan.md Phase 4): training-data completeness has never been
checked directly before, only sampled. Streams the FULL SFT and RL reasoning corpora (not a subset)
and counts:

  - rows with a non-empty `reasoning` trace
  - rows with each reasoned-context column present (interpro_formatted, ppi_formatted,
    subcellular_location)
  - rows whose ground truth is EMPTY per aspect (go_mf / go_bp / go_cc)
  - rows with a non-empty `final_answer` and a "known" `protein_function` (plan.md Phase 4 / gate
    R3: the functional summary that `bioreason_pro.data_contract.compose_functional_summary` now
    splices into the SFT answer for `leaf_only_reasoned` -- this measures what fraction of SFT rows
    can actually supervise it, now that a missing `final_answer` skips the row (MissingReasoningEvidence)
  - confirms the RL corpus's `reasoning` column is absent from the schema entirely (ADR-026), not
    merely empty -- a schema fact, checked directly rather than re-sampled

Read-only, no training, no licence-status change: this only counts columns already approved for
sft-training/rl-training use, under `use="sft-training"`/`use="rl-training"` exactly as those stages
already read them -- final_answer/protein_function counting reflects gate R3's 2026-08-23 adoption.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.data_contract import _has_known_protein_function, parse_go_terms  # noqa: E402
from bioreason_pro.license_policy import require_approved_dataset  # noqa: E402

CONTEXT_COLUMNS = ("interpro_formatted", "ppi_formatted", "subcellular_location")
GT_COLUMNS = ("go_mf", "go_bp", "go_cc")


def _stream(repo_id: str, use: str):
    from datasets import load_dataset

    revision = require_approved_dataset(repo_id, use)
    return load_dataset(repo_id, split="train", streaming=True, revision=revision)


def summarize_rows(rows) -> dict:
    """Pure counting pass over an iterable of raw corpus rows (unit-tested without network)."""
    n = 0
    reasoning_nonempty = 0
    final_answer_nonempty = 0
    protein_function_known = 0
    context_nonempty = Counter()
    gt_empty = Counter()
    schema_columns: list[str] | None = None
    for row in rows:
        if not row.get("protein_id") or not row.get("sequence"):
            continue
        if schema_columns is None:
            schema_columns = sorted(row.keys())
        n += 1
        if str(row.get("reasoning") or "").strip():
            reasoning_nonempty += 1
        if str(row.get("final_answer") or "").strip():
            final_answer_nonempty += 1
        if _has_known_protein_function(row.get("protein_function")):
            protein_function_known += 1
        for col in CONTEXT_COLUMNS:
            if str(row.get(col) or "").strip():
                context_nonempty[col] += 1
        for col in GT_COLUMNS:
            if not parse_go_terms(row.get(col)):
                gt_empty[col] += 1
    return {
        "rows": n,
        "has_reasoning_column": "reasoning" in (schema_columns or []),
        "reasoning_nonempty": reasoning_nonempty,
        "reasoning_nonempty_frac": reasoning_nonempty / n if n else 0.0,
        "has_final_answer_column": "final_answer" in (schema_columns or []),
        "final_answer_nonempty": final_answer_nonempty,
        "final_answer_nonempty_frac": final_answer_nonempty / n if n else 0.0,
        "has_protein_function_column": "protein_function" in (schema_columns or []),
        "protein_function_known": protein_function_known,
        "protein_function_known_frac": protein_function_known / n if n else 0.0,
        "context_nonempty": dict(context_nonempty),
        "context_nonempty_frac": {c: context_nonempty[c] / n if n else 0.0 for c in CONTEXT_COLUMNS},
        "gt_empty": dict(gt_empty),
        "gt_empty_frac": {c: gt_empty[c] / n if n else 0.0 for c in GT_COLUMNS},
    }


def audit_corpus(repo_id: str, use: str) -> dict:
    n = 0

    def _logged_rows():
        nonlocal n
        for row in _stream(repo_id, use):
            n += 1
            if n % 20000 == 0:
                print(f"[audit] {repo_id}: streamed {n} rows", file=sys.stderr)
            yield row

    return {"repo_id": repo_id, **summarize_rows(_logged_rows())}


def main() -> int:
    results = {}
    print("=== SFT corpus (wanglab/bioreason-pro-sft-reasoning-data) ===", file=sys.stderr)
    results["sft"] = audit_corpus("wanglab/bioreason-pro-sft-reasoning-data", "sft-training")
    print("=== RL corpus (wanglab/bioreason-pro-rl-reasoning-data) ===", file=sys.stderr)
    results["rl"] = audit_corpus("wanglab/bioreason-pro-rl-reasoning-data", "rl-training")

    out_path = ROOT / "outputs" / "phase4_corpus_completeness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for label, r in results.items():
        print(f"\n[{label}] {r['repo_id']}: {r['rows']} rows")
        print(f"  reasoning column present : {r['has_reasoning_column']}")
        print(f"  reasoning non-empty      : {r['reasoning_nonempty']} "
              f"({r['reasoning_nonempty_frac'] * 100:.1f}%)")
        print(f"  final_answer column present: {r['has_final_answer_column']}")
        print(f"  final_answer non-empty   : {r['final_answer_nonempty']} "
              f"({r['final_answer_nonempty_frac'] * 100:.1f}%)")
        print(f"  protein_function column present: {r['has_protein_function_column']}")
        print(f"  protein_function known   : {r['protein_function_known']} "
              f"({r['protein_function_known_frac'] * 100:.1f}%)")
        for c in CONTEXT_COLUMNS:
            print(f"  {c:22s} non-empty: {r['context_nonempty'].get(c, 0)} "
                  f"({r['context_nonempty_frac'][c] * 100:.1f}%)")
        for c in GT_COLUMNS:
            print(f"  {c} empty             : {r['gt_empty'].get(c, 0)} "
                  f"({r['gt_empty_frac'][c] * 100:.1f}%)")
    print(f"\n[audit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
