"""Offline tests for the Weave Evaluation scorers.

The scorers are pure functions so they can be tested without a Weave server. What matters most is
the boundary they must not cross: none of them may present itself as `weighted_fmax`, which only
the protected global scorer produces.
"""

from __future__ import annotations

import pytest

from bioreason_pro import weave_eval as WE

# A -> B -> C chain plus an unrelated term.
ANCESTORS = {
    "GO:0000003": {"GO:0000002", "GO:0000001"},
    "GO:0000002": {"GO:0000001"},
    "GO:0000001": set(),
    "GO:0000009": set(),
}
IA = {"GO:0000001": 0.0, "GO:0000002": 2.0, "GO:0000003": 5.0, "GO:0000009": 3.0}


def test_coverage_scorer_separates_empty_from_populated():
    assert WE.score_coverage("<think>x</think> GO:0000003") == {
        WE.COVERAGE_KEY: True,
        "diag_n_terms": 1,
    }
    assert WE.score_coverage("<think>x</think> no identifier") == {
        WE.COVERAGE_KEY: False,
        "diag_n_terms": 0,
    }


def test_redundancy_scorer_measures_the_wasted_share():
    # All three of the chain are emitted; only the leaf is not implied by another.
    response = "GO:0000001 GO:0000002 GO:0000003"
    out = WE.score_ancestor_redundancy(response, ANCESTORS)
    assert out["diag_n_specific_terms"] == 1
    assert out["diag_redundant_fraction"] == pytest.approx(2 / 3)


def test_redundancy_scorer_is_zero_for_a_leaf_only_answer():
    out = WE.score_ancestor_redundancy("GO:0000003 GO:0000009", ANCESTORS)
    assert out["diag_redundant_fraction"] == 0.0
    assert out["diag_n_specific_terms"] == 2


def test_redundancy_scorer_handles_an_empty_prediction():
    out = WE.score_ancestor_redundancy("nothing here", ANCESTORS)
    assert out == {"diag_redundant_fraction": 0.0, "diag_n_specific_terms": 0}


def test_ia_f1_scorer_is_perfect_when_the_propagated_sets_match():
    out = WE.score_ia_weighted_f1("GO:0000003", ["GO:0000003"], ANCESTORS, IA)
    assert out[WE.IA_F1_KEY] == 1.0


def test_ia_f1_scorer_is_zero_without_a_prediction():
    out = WE.score_ia_weighted_f1("no identifier", ["GO:0000003"], ANCESTORS, IA)
    assert out[WE.IA_F1_KEY] == 0.0


def test_no_scorer_key_can_be_mistaken_for_the_metric_of_record():
    """The global swept F_max is not per-example; nothing here may claim that name."""
    keys = set()
    keys |= set(WE.score_coverage("GO:0000003"))
    keys |= set(WE.score_ia_weighted_f1("GO:0000003", ["GO:0000003"], ANCESTORS, IA))
    keys |= set(WE.score_ancestor_redundancy("GO:0000003", ANCESTORS))
    assert keys, "expected scorer keys"
    for key in keys:
        assert key.startswith("diag_"), key
        assert "weighted_fmax" not in key or key == WE.IA_F1_KEY
    assert "weighted_fmax" not in keys


def test_dataset_rows_are_stable_and_carry_ground_truth():
    records = [
        {"protein_id": "P2", "generated_response": "GO:0000003", "gt_terms": {"GO:0000009", "GO:0000003"}},
        {"protein_id": "P1", "generated_response": "", "gt_terms": set()},
    ]
    rows = WE.build_dataset(records)
    assert rows == [
        {"protein_id": "P2", "gt_terms": ["GO:0000003", "GO:0000009"]},
        {"protein_id": "P1", "gt_terms": []},
    ]


def test_publish_is_best_effort_and_never_raises(monkeypatch):
    """A Weave outage must not fail an eval run whose metric of record already exists."""
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "weave":
            raise RuntimeError("weave is unreachable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    url = WE.publish_evaluation(
        [{"protein_id": "P1", "generated_response": "GO:0000003", "gt_terms": {"GO:0000003"}}],
        {"weighted_fmax": 0.5},
        obo_ancestors=ANCESTORS,
        ia=IA,
        name="unit",
    )
    assert url is None


def test_in_loop_val_does_not_publish_but_the_standalone_runner_does():
    """Wiring check at source level: _generation_eval needs a GPU.

    In-loop val runs on every eval interval, so publishing there would add round trips to the
    training loop; the standalone A/B runner is where per-protein inspection is wanted.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    train_src = (root / "train.py").read_text(encoding="utf-8")
    runner_src = (root / "scripts" / "compare_prompt_variants.py").read_text(encoding="utf-8")

    assert "publish_weave_eval: bool = False" in train_src
    assert "publish_weave_eval=True" in runner_src
    # The authoritative metric must be attached, never recomputed from scorers.
    assert "metric_of_record" in (root / "bioreason_pro" / "weave_eval.py").read_text(
        encoding="utf-8"
    )


def test_publish_initialises_weave_rather_than_assuming_a_client(monkeypatch):
    """Job 630 published nothing because no weave.init had run; the drop was silent."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "bioreason_pro" / "weave_eval.py").read_text(
        encoding="utf-8"
    )
    assert "weave.init(weave_project)" in src
    # The runner initialises too, matching train.py's tracking setup.
    runner = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_prompt_variants.py"
    ).read_text(encoding="utf-8")
    assert "weave.init(WEAVE_PROJECT)" in runner


def test_failure_is_reported_loudly_not_as_a_missing_url():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "bioreason_pro" / "weave_eval.py").read_text(
        encoding="utf-8"
    )
    assert "[weave-eval] FAILED" in src
    assert "[weave-eval] published" in src


def test_evaluation_call_display_name_is_set_explicitly():
    """`Evaluation(name=...)` names the object, not the call.

    Without setting the call's display name, every row in the Evals view gets an auto-generated name
    like "eval-2026-08-11-joyful-plateau" and nothing identifies which job, checkpoint, budget or arm
    produced it. 51 evaluations from this search landed that way before this was fixed. Verified
    against the live project: a probe published as "NAME-CHECK-display-name-fix" and appeared under
    that name.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "bioreason_pro" / "weave_eval.py").read_text(
        encoding="utf-8"
    )
    assert "call.set_display_name(name)" in src
    # `.call()` on a bound method op does not bind self.
    assert "evaluation.evaluate.call(evaluation, model)" in src
