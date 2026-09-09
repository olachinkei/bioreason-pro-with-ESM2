"""Weave Evaluation for managing eval runs (AUTHORED).

`weave_tracing.py` covers RL rollouts; this covers evaluation. It publishes each eval run as a
`weave.Evaluation` so runs are comparable side by side in the Weave UI and every prediction is
inspectable per protein.

## The metric of record is NOT recomputed here

`weighted_fmax` is a threshold-swept, IA-weighted F_max that `cafaeval` computes **globally** over
the whole prediction set: it sweeps one decision threshold across all proteins and takes the best
operating point per aspect. It therefore does not decompose into per-example contributions, and any
per-example scorer that claimed to produce it would be a different, quietly-wrong number.

So the split is deliberate:

* **Scorers** here report per-example diagnostics — did this protein get a prediction, how many
  terms, how specific were they, what is its own IA-weighted F1. These are what you read when
  searching for conditions that improve the model.
* **The authoritative `weighted_fmax`** is computed once by the protected `eval.score_generations`
  and attached to the Evaluation as attributes, so the Weave record points at the real number
  instead of inventing a second one.

Per-example `ia_weighted_f1` is genuinely useful but is *not* the metric of record: it is a
per-sample mean, not a swept global F_max. The two move together but are not equal, and the keys
are named so they cannot be confused.
"""

from __future__ import annotations

from typing import Any

from bioreason_pro.rewards import (
    extract_go_terms,
    ia_weighted_f1,
    propagate_ancestors,
    strip_implied_ancestors,
)

# Scorer keys are prefixed so nothing here can be mistaken for `weighted_fmax`.
COVERAGE_KEY = "diag_has_prediction"
IA_F1_KEY = "diag_ia_weighted_f1_per_sample"


def score_coverage(generated_response: str, final_answer_only: bool = False) -> dict[str, Any]:
    """Did this protein produce any GO id, and how many?

    A protein with no id is dropped from the prediction file but still counted in the recall
    denominator (cafaeval's `norm='cafa'`), so this is the first thing to look at when the global
    metric is low.
    """
    terms = extract_go_terms(generated_response, final_answer_only=final_answer_only)
    return {COVERAGE_KEY: bool(terms), "diag_n_terms": len(terms)}


def score_ia_weighted_f1(
    generated_response: str,
    gt_terms,
    obo_ancestors: dict[str, set[str]],
    ia: dict[str, float] | None,
    final_answer_only: bool = False,
) -> dict[str, Any]:
    """Per-sample IA-weighted F1 over ancestor-propagated sets. NOT the swept global F_max."""
    predicted = propagate_ancestors(
        extract_go_terms(generated_response, final_answer_only=final_answer_only), obo_ancestors
    )
    truth = propagate_ancestors(set(gt_terms or ()), obo_ancestors)
    return {IA_F1_KEY: ia_weighted_f1(predicted, truth, ia)}


def score_ancestor_redundancy(
    generated_response: str,
    obo_ancestors: dict[str, set[str]],
    final_answer_only: bool = False,
) -> dict[str, Any]:
    """Share of emitted terms that propagation would have added anyway.

    Scoring propagates predictions, so an emitted ancestor buys nothing. Under a fixed generation
    budget this fraction is budget spent for no score.
    """
    terms = extract_go_terms(generated_response, final_answer_only=final_answer_only)
    if not terms:
        return {"diag_redundant_fraction": 0.0, "diag_n_specific_terms": 0}
    specific = strip_implied_ancestors(terms, obo_ancestors)
    return {
        "diag_redundant_fraction": 1.0 - len(specific) / len(terms),
        "diag_n_specific_terms": len(specific),
    }


def build_dataset(records) -> list[dict[str, Any]]:
    """Weave dataset rows. `gt_terms` is sorted so rows hash stably across runs."""
    return [
        {"protein_id": r["protein_id"], "gt_terms": sorted(r.get("gt_terms") or ())}
        for r in records
    ]


WEAVE_PROJECT = "wandb-healthcare/bioreasonpro-senpai"


def publish_evaluation(
    records,
    metrics: dict[str, float],
    *,
    obo_ancestors: dict[str, set[str]],
    ia: dict[str, float] | None,
    name: str,
    attributes: dict[str, Any] | None = None,
    final_answer_only: bool = False,
    weave_project: str = WEAVE_PROJECT,
) -> str | None:
    """Publish one eval run as a weave.Evaluation. Returns a locator string, or None on failure.

    Best-effort by design: evaluation *reporting* must never take down an eval run whose metric of
    record has already been computed. `metrics` is that authoritative result and is attached as
    attributes rather than recomputed.

    `weave.init` is called here rather than assumed. Without an initialised client the Evaluation is
    dropped inside Weave's background executor, which surfaces as a logged ERROR rather than an
    exception — so the caller sees neither a URL nor a failure and believes it published.
    """
    try:
        import asyncio

        import weave

        # Idempotent: re-initialising an already-initialised project is a no-op.
        weave.init(weave_project)

        predictions = {r["protein_id"]: r["generated_response"] for r in records}
        dataset = build_dataset(records)

        @weave.op()
        def model(protein_id: str) -> str:
            """Replay the stored generation for this protein (no model is loaded here)."""
            return predictions.get(protein_id, "")

        @weave.op()
        def coverage_scorer(output: str) -> dict[str, Any]:
            return score_coverage(output, final_answer_only)

        @weave.op()
        def ia_f1_scorer(gt_terms, output: str) -> dict[str, Any]:
            return score_ia_weighted_f1(output, gt_terms, obo_ancestors, ia, final_answer_only)

        @weave.op()
        def redundancy_scorer(output: str) -> dict[str, Any]:
            return score_ancestor_redundancy(output, obo_ancestors, final_answer_only)

        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[coverage_scorer, ia_f1_scorer, redundancy_scorer],
            name=name,
        )
        # The metric of record travels with the Evaluation instead of being recomputed from scorers.
        merged = {
            "metric_of_record": {k: v for k, v in metrics.items() if not k.startswith("diag/")},
            "diagnostics": {k: v for k, v in metrics.items() if k.startswith("diag/")},
            **(attributes or {}),
        }
        # `Evaluation(name=...)` names the OBJECT; the CALL gets an auto-generated display name like
        # "eval-2026-08-11-joyful-plateau" unless it is set explicitly. Without this every row in the
        # Evals view is unidentifiable — which job, checkpoint, budget or arm produced it is lost, and
        # comparing conditions there is impossible. Same `.call()` + set_display_name pattern
        # weave_tracing.py already uses for rollouts.
        async def _evaluate_named():
            # `.call()` on a bound method op does not bind self — pass the Evaluation explicitly.
            result, call = await evaluation.evaluate.call(evaluation, model)
            call.set_display_name(name)
            return result

        with weave.attributes(merged):
            summary = asyncio.run(_evaluate_named())
        # `Evaluation` exposes no ui_url, so report a locator that is always printable: an empty
        # return value previously made a silent no-op indistinguishable from a successful publish.
        locator = f"https://wandb.ai/{weave_project}/weave/evaluations name={name}"
        print(f"[weave-eval] published {len(dataset)} examples; summary_keys="
              f"{sorted(summary or {})}", flush=True)
        return locator
    except Exception as exc:  # pragma: no cover - network/version dependent
        print(f"[weave-eval] FAILED ({type(exc).__name__}: {exc})", flush=True)
        return None
