"""Thin wrapper over the `cafaeval` PyPI package (AUTHORED). PROTECTED.

Mirrors the AUTHORITATIVE public usage in bioreason2 evals/cafa_evals.py
(run_cafa_evaluation + extract_metrics_summary):

    from cafaeval.evaluation import cafa_eval
    evaluation_df, best_scores_dict = cafa_eval(obo_file, pred_dir, gt_file, ia_file, th_step=...)

- `best_scores_dict` is a DICT of DataFrames keyed by metric ("f", "f_w", ...); each frame has
  columns `f` (F1) and `f_w` (IA-weighted F1) and an `ns` (namespace) index/column.
- Per-aspect weighted F_max = the `f_w` value per `ns`; there is NO built-in "overall" row —
  the top-line is the caller-computed mean over MF/BP/CC (aggregate_weighted_fmax).
- `pred_dir` is a DIRECTORY of prediction TSVs (target_id, GO_term, score). The public pipeline
  writes score=1.0 for every predicted term, so a coarse th_step suffices for that setup.
"""

from __future__ import annotations

# cafaeval namespace labels -> our aspect keys
NS_TO_ASPECT = {
    "molecular_function": "mf", "biological_process": "bp", "cellular_component": "cc",
    "MFO": "mf", "BPO": "bp", "CCO": "cc",  # tolerate alternate labels
}


def aggregate_weighted_fmax(per_aspect: dict[str, float]) -> dict[str, float]:
    """Given {'mf': .., 'bp': .., 'cc': ..}, return those plus the overall mean.

    Pure function (no cafaeval dependency) so it is unit-testable. Averages only present aspects.
    """
    aspects = {a: v for a, v in per_aspect.items() if a in ("mf", "bp", "cc") and v is not None}
    out = {f"weighted_fmax_{a}": v for a, v in aspects.items()}
    out["weighted_fmax"] = sum(aspects.values()) / len(aspects) if aspects else float("nan")
    # HAZARD: the mean is over the aspects cafaeval RETURNED, and it returns nothing for a namespace
    # with no predictions at all. A model that predicts only MF is therefore scored on MF alone — a
    # smaller denominator and a higher number that is not comparable to a three-aspect mean. An
    # aspect_mean-trained checkpoint decoded at 16 tokens emits one MF term and scores 0.477 this way,
    # against 0.278 for the same checkpoint at 64 tokens where all three aspects appear.
    #
    # The metric of record is deliberately left alone. What is added is the denominator, so any
    # comparison can see it: an arm with n_aspects < 3 must never be ranked against one with 3.
    out["weighted_fmax_n_aspects"] = float(len(aspects))
    return out


def _per_aspect_fw(best_scores_dict) -> dict[str, float]:
    """Extract {aspect: weighted-F_max} from cafa_eval's best_scores_dict (pure given the dict)."""
    df = best_scores_dict.get("f_w", best_scores_dict.get("f"))
    if df is None:
        raise KeyError("cafa_eval returned no 'f_w'/'f' best-scores frame — pass the IA file (ia).")
    df = df.reset_index()
    if "f_w" not in df.columns:
        raise KeyError("best-scores frame has no 'f_w' column — IA weights were not supplied.")
    out: dict[str, float] = {}
    for ns in df["ns"].unique():
        aspect = NS_TO_ASPECT.get(ns)
        if aspect:
            out[aspect] = float(df[df["ns"] == ns].iloc[0]["f_w"])
    return out


def weighted_fmax(obo_file: str, pred_dir: str, gt_file: str, ia_file: str,
                  th_step: float = 0.1) -> dict[str, float]:
    """Run cafa_eval and return {'weighted_fmax', 'weighted_fmax_{mf,bp,cc}'} (aspect-mean overall).

    `ia_file` is passed positionally (matching evals/cafa_evals.run_cafa_evaluation). Use a smaller
    th_step for real varied confidence scores; the public score=1.0 pipeline uses a coarse step.
    """
    from cafaeval.evaluation import cafa_eval  # imported lazily; cafaeval is a base dep

    _evaluation_df, best_scores_dict = cafa_eval(obo_file, pred_dir, gt_file, ia_file, th_step=th_step)
    return aggregate_weighted_fmax(_per_aspect_fw(best_scores_dict))
