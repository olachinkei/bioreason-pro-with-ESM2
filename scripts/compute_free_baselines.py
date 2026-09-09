#!/usr/bin/env python3
"""Compute plan.md Phase 2's two free baselines — a zero-parameter `interpro2go` lookup and a
label-prior — scored by the same `eval.score_generations` path every model arm uses, on both the old
val-256 subset and the Phase 1 `cafa_no_knowledge` dev set.

Prerequisites:
    python scripts/fetch_approved_reference_data.py
    python scripts/fetch_interpro2go_mapping.py
    python scripts/build_cafa_no_knowledge_dev_set.py   # already run for Phase 1

Shared-dependency note (plan.md Phase 2): the interpro2go lookup needs `interpro_formatted`, and so
does the reasoned model arm's prompt. `cafa_no_knowledge` has neither until Phase 1's InterProScan job
completes. This script does NOT fall back to scoring interpro2go on the old split while labelling it
as the new one's number — it reports the interpro2go row as unavailable on `cafa_no_knowledge` rather
than silently substituting.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.baselines import (  # noqa: E402
    as_generated_response,
    extract_interpro_ids,
    interpro2go_predict,
    label_prior_terms,
    parse_interpro2go_mapping,
)
from bioreason_pro.license_policy import require_approved_dataset  # noqa: E402

SFT_DATA_REPO = "wanglab/bioreason-pro-sft-reasoning-data"
ASPECTS = ("go_mf", "go_bp", "go_cc")
VAL_SUBSET_SIZE = 256
VAL_SEED = 0
# Bracketed on both sides (rule 2): the smallest and largest values below are both worse than the
# interior peak in every run so far, so the search is not sitting at an edge.
LABEL_PRIOR_N_CANDIDATES = (1, 2, 5, 10, 20, 40, 80)


def _parse_go(v) -> set[str]:
    from bioreason_pro.data_contract import parse_go_terms

    return parse_go_terms(v)


def _stream_corpus_rows():
    """Raw rows of the SFT corpus under the Phase 2 'baseline-evaluation' use (see approved_assets.json:
    this use additionally reads `interpro_formatted`, never injected into a prompt or training run)."""
    from datasets import load_dataset

    revision = require_approved_dataset(SFT_DATA_REPO, "baseline-evaluation")
    return load_dataset(SFT_DATA_REPO, split="train", streaming=True, revision=revision)


def _collect_val_subset_and_term_counts():
    """One streaming pass over the full corpus: the val-256 rows (with interpro_formatted + GT) for
    the interpro2go arm, and full-corpus per-aspect term frequency counts for the label-prior arm."""
    import data

    val_ids: list[str] = []
    val_rows_by_id: dict[str, dict] = {}
    term_counts = {aspect: Counter() for aspect in ASPECTS}

    n = 0
    for row in _stream_corpus_rows():
        pid, seq = row.get("protein_id"), row.get("sequence")
        if not pid or not seq:
            continue
        n += 1
        for aspect in ASPECTS:
            term_counts[aspect].update(_parse_go(row.get(aspect)))
        if data.in_split(pid, "val", VAL_SEED):
            val_ids.append(pid)
            val_rows_by_id[pid] = row
        if n % 20000 == 0:
            print(f"[baselines] streamed {n} corpus rows, {len(val_ids)} val so far", file=sys.stderr)

    subset_ids = data.deterministic_val_subset(val_ids, size=VAL_SUBSET_SIZE, seed=VAL_SEED)
    val_subset_rows = [val_rows_by_id[pid] for pid in subset_ids]
    print(f"[baselines] {n} corpus rows, {len(val_ids)} val, {len(val_subset_rows)}-protein val subset",
          file=sys.stderr)
    return val_subset_rows, term_counts


def _load_cafa_no_knowledge_rows() -> list[dict]:
    path = ROOT / "data" / "cafa_no_knowledge_dev_set.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _row_gt(row: dict) -> set[str]:
    gt: set[str] = set()
    for aspect in ASPECTS:
        gt |= _parse_go(row.get(aspect))
    return gt


def _score(records: list[dict], out_dir: Path) -> dict[str, float]:
    import eval as sealed_eval

    return sealed_eval.score_generations(records, str(out_dir), diagnostics=True)


def interpro2go_arm(rows: list[dict], mapping: dict[str, set[str]], has_interpro: bool,
                    out_dir: Path) -> dict[str, float] | None:
    if not has_interpro:
        print("[baselines] interpro2go: SKIPPED — no interpro_formatted available on this split",
              file=sys.stderr)
        return None
    records = []
    for row in rows:
        ipr_ids = extract_interpro_ids(row.get("interpro_formatted"))
        predicted = interpro2go_predict(ipr_ids, mapping)
        records.append({"protein_id": row["protein_id"],
                         "generated_response": as_generated_response(predicted),
                         "gt_terms": _row_gt(row)})
    metrics = _score(records, out_dir)
    print(f"[baselines] interpro2go: weighted_fmax={metrics['weighted_fmax']:.5f} "
          f"n_aspects={metrics.get('weighted_fmax_n_aspects')} "
          f"coverage={metrics.get('diag/coverage'):.3f}", file=sys.stderr)
    return metrics


def label_prior_arm(rows: list[dict], term_counts: dict[str, Counter], out_dir: Path) -> dict:
    """Sweep top_n (rule 1/2: sweep and bracket), return {n: metrics} plus the peak."""
    swept = {}
    for n in LABEL_PRIOR_N_CANDIDATES:
        prior = label_prior_terms(term_counts, top_n=dict.fromkeys(ASPECTS, n))
        predicted = set().union(*prior.values())
        records = [{"protein_id": row["protein_id"],
                    "generated_response": as_generated_response(predicted),
                    "gt_terms": _row_gt(row)} for row in rows]
        swept[n] = _score(records, out_dir / f"n{n}")
        print(f"[baselines] label_prior n={n}: weighted_fmax={swept[n]['weighted_fmax']:.5f} "
              f"n_aspects={swept[n].get('weighted_fmax_n_aspects')}", file=sys.stderr)
    peak_n = max(swept, key=lambda n: swept[n]["weighted_fmax"])
    if peak_n in (LABEL_PRIOR_N_CANDIDATES[0], LABEL_PRIOR_N_CANDIDATES[-1]):
        print(f"[baselines] WARNING: label_prior peak n={peak_n} sits at the swept edge — "
              "extend LABEL_PRIOR_N_CANDIDATES before reporting this as bracketed (rule 2)",
              file=sys.stderr)
    return {"swept": swept, "peak_n": peak_n, "peak": swept[peak_n]}


def main() -> int:
    out_root = ROOT / "outputs" / "phase2_baselines"
    out_root.mkdir(parents=True, exist_ok=True)

    with (ROOT / "data" / "interpro2go.txt").open(encoding="utf-8") as handle:
        mapping = parse_interpro2go_mapping(handle)
    print(f"[baselines] interpro2go mapping: {len(mapping)} IPR ids", file=sys.stderr)

    val_rows, term_counts = _collect_val_subset_and_term_counts()
    cafa_rows = _load_cafa_no_knowledge_rows()

    results: dict[str, dict] = {}

    print("\n=== old val-256 subset ===", file=sys.stderr)
    val_has_interpro = any(r.get("interpro_formatted") for r in val_rows)
    results["val_interpro2go"] = interpro2go_arm(
        val_rows, mapping, val_has_interpro, out_root / "val_interpro2go"
    )
    results["val_label_prior"] = label_prior_arm(val_rows, term_counts, out_root / "val_label_prior")

    print("\n=== cafa_no_knowledge (1,496 proteins) ===", file=sys.stderr)
    cafa_has_interpro = any(r.get("interpro_formatted") for r in cafa_rows)
    results["cafa_no_knowledge_interpro2go"] = interpro2go_arm(
        cafa_rows, mapping, cafa_has_interpro, out_root / "cafa_interpro2go"
    )
    results["cafa_no_knowledge_label_prior"] = label_prior_arm(
        cafa_rows, term_counts, out_root / "cafa_label_prior"
    )

    (out_root / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[baselines] wrote {out_root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
