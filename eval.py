"""eval.py — PROTECTED sealed evaluator. The ONLY approved path to the metric.

Mirrors the authoritative public pipeline (bioreason2 evals/cafa_evals.py): extract GO terms from
generations, write CAFA-format TSVs (target_id, GO_term, score) into a directory, then score with
`cafaeval` via bioreason_pro.cafa_fmax (threshold-swept IA-weighted F_max, aspect-mean overall).
A reviewer re-runs THIS independently on the merged checkpoint artifact — never a self-reported number.

The TSV writing + scoring are authored + integration-tested against the real cafaeval; only model
generation (needs the checkpoint + GPU) is the remaining TODO in `evaluate`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bioreason_pro.cafa_fmax import weighted_fmax
from bioreason_pro.rewards import extract_go_terms

OBO = "data/go-basic.obo"
IA = "data/IA.txt"


def write_cafa_predictions(predictions, pred_dir: str, filename: str = "predictions.tsv") -> str:
    """Write CAFA prediction TSV(s) into a DIRECTORY (cafa_eval walks it). Returns the dir.

    `predictions`: iterable of (target_id, terms), where terms is a {GO_term: score} mapping or a
    set/list of GO_terms (score defaults to 1.0, matching the public pipeline's binary scoring).
    """
    out = Path(pred_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for target_id, terms in predictions:
        items = terms.items() if isinstance(terms, dict) else ((t, 1.0) for t in terms)
        for go_term, score in items:
            lines.append(f"{target_id}\t{go_term}\t{score}")
    (out / filename).write_text("\n".join(lines) + ("\n" if lines else ""))
    return str(out)


def write_cafa_ground_truth(ground_truth, gt_file: str) -> str:
    """Write the CAFA ground-truth TSV (target_id, GO_term). `ground_truth`: (target_id, terms)."""
    lines = []
    for target_id, terms in ground_truth:
        for go_term in terms:
            lines.append(f"{target_id}\t{go_term}")
    Path(gt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(gt_file).write_text("\n".join(lines) + ("\n" if lines else ""))
    return gt_file


def prediction_diagnostics(records, final_answer_only: bool = False) -> dict[str, float]:
    """Shape statistics for a set of generations. DIAGNOSTIC ONLY — never a selection signal.

    `cafa_eval` runs with `norm='cafa'`: precision is averaged over proteins that received at least
    one prediction, while recall is averaged over every ground-truth protein. A protein whose
    generation yields no `GO:` id is therefore not neutral — it is dropped from `predictions` by
    `score_generations` and still counted in the recall denominator. Coverage is consequently a
    first-order driver of weighted F_max, and nothing in the pipeline reported it.

    `recall_ceiling_from_coverage` is the exact upper bound this imposes: even with a perfect
    prediction on every covered protein, unweighted recall cannot exceed the share of ground-truth
    terms that sit on covered proteins.

    `think_only_term_fraction` measures the opposite failure. Terms are extracted from the WHOLE
    response by default, so GO ids a thinking model enumerates while reasoning become predictions
    and cost precision. This is the size of that effect, i.e. what `final_answer_only=True` drops.
    """
    n_records = n_covered = n_covered_final = 0
    pred_terms = gt_terms_total = gt_terms_covered = 0
    think_only_terms = 0
    for record in records:
        n_records += 1
        response = record["generated_response"]
        predicted = extract_go_terms(response, final_answer_only=final_answer_only)
        final_only = extract_go_terms(response, final_answer_only=True)
        gt = set(record.get("gt_terms") or ())
        gt_terms_total += len(gt)
        pred_terms += len(predicted)
        think_only_terms += len(predicted - final_only)
        if predicted:
            n_covered += 1
            gt_terms_covered += len(gt)
        if final_only:
            n_covered_final += 1
    return {
        "n_records": float(n_records),
        "n_covered": float(n_covered),
        "coverage": n_covered / n_records if n_records else 0.0,
        "coverage_final_answer_only": n_covered_final / n_records if n_records else 0.0,
        "recall_ceiling_from_coverage": (
            gt_terms_covered / gt_terms_total if gt_terms_total else 0.0
        ),
        "pred_terms_per_protein": pred_terms / n_records if n_records else 0.0,
        "pred_terms_per_covered_protein": pred_terms / n_covered if n_covered else 0.0,
        "gt_terms_per_protein": gt_terms_total / n_records if n_records else 0.0,
        "think_only_term_fraction": think_only_terms / pred_terms if pred_terms else 0.0,
    }


def score_generations(records, out_dir: str, obo: str = OBO, ia: str = IA,
                      final_answer_only: bool = False, th_step: float = 0.1,
                      diagnostics: bool = False) -> dict[str, float]:
    """Extract predictions from generations, write CAFA TSVs, and score with cafaeval.

    `records`: iterable of {"protein_id", "generated_response", "gt_terms": set[str]}.
    Returns {'weighted_fmax', 'weighted_fmax_{mf,bp,cc}'} (aspect-mean overall).

    `diagnostics=True` additionally returns `prediction_diagnostics` under a `diag/` prefix. It is
    off by default so the sealed-test path keeps emitting exactly the metric of record; the metric
    itself is unchanged either way.
    """
    if obo == OBO and ia == IA:
        from bioreason_pro.license_policy import validate_reference_assets

        validate_reference_assets(use="evaluation")
    records = list(records)
    predictions, ground_truth = [], []
    for r in records:
        pid = r["protein_id"]
        pred = extract_go_terms(r["generated_response"], final_answer_only=final_answer_only)
        if pred:
            predictions.append((pid, pred))
        if r.get("gt_terms"):
            ground_truth.append((pid, set(r["gt_terms"])))
    out = Path(out_dir)
    pred_dir = write_cafa_predictions(predictions, str(out / "predictions"))
    gt_file = write_cafa_ground_truth(ground_truth, str(out / "ground_truth.tsv"))
    metrics = weighted_fmax(obo, pred_dir, gt_file, ia, th_step=th_step)
    if diagnostics:
        diag = prediction_diagnostics(records, final_answer_only=final_answer_only)
        metrics = {**metrics, **{f"diag/{k}": v for k, v in diag.items()}}
    return metrics


def evaluate(checkpoint, split: str = "val", subset_size: int | None = 256,
             out_dir: str = "outputs/eval", target: str = "bioreason_pro_test",
             diagnostics: bool = False) -> dict[str, float]:
    """Run the model over the held-out `target` and return the weighted-F_max metrics.

    split='val' with subset_size=256 → the deterministic in-loop steering metric.
    split='test' with subset_size=None → the sealed final claim metric (milestone/best-ckpt only).
    `target` selects the held-out dataset (see the editable eval_targets/ package): 'bioreason_pro_test'
    (available) or 'cafa5' (pending HF access).

    Model generation is delegated to the editable train.run_sealed_eval (checkpoint assembly + protein-
    aware fused decode); the metric scoring stays HERE (score_generations → cafa_fmax.weighted_fmax),
    so the metric definition is unchanged.
    """
    import train  # editable; import is side-effect-free (train's heavy deps are lazy)

    records = train.run_sealed_eval(checkpoint, target=target, split=split, subset_size=subset_size)
    return score_generations(records, out_dir, diagnostics=diagnostics)


def _resolve_cli_subset_size(split: str, subset_size: int | None) -> int | None:
    """Keep validation bounded by default while making test subsets explicit and test-full default."""
    if subset_size is not None and subset_size < 1:
        raise ValueError("--subset_size must be a positive integer")
    if split == "val" and subset_size is None:
        return 256
    return subset_size


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="explicit bounded subset; omit with --split test to evaluate the full sealed holdout",
    )
    p.add_argument("--target", default="bioreason_pro_test",
                   help="held-out dataset (eval_targets/): bioreason_pro_test | cafa5")
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="also report coverage/prediction-shape diagnostics (diag/*); never a selection signal",
    )
    args = p.parse_args()
    subset = _resolve_cli_subset_size(args.split, args.subset_size)
    metrics = evaluate(
        args.checkpoint, args.split, subset, target=args.target, diagnostics=args.diagnostics
    )
    print({f"{args.split}_primary/{k}": v for k, v in metrics.items()})


if __name__ == "__main__":
    _cli()
