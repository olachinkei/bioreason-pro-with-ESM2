"""Offline tests for the coverage / prediction-shape diagnostics in eval.py."""

from __future__ import annotations

import pytest

import eval as E

try:
    from bioreason_pro.license_policy import validate_reference_assets

    validate_reference_assets(use="evaluation")
    _HAVE_DATA = True
except Exception:
    _HAVE_DATA = False

needs_reference_data = pytest.mark.skipif(
    not _HAVE_DATA, reason="approved go-basic.obo + IA.txt are not installed"
)


def _record(pid, response, gt):
    return {"protein_id": pid, "generated_response": response, "gt_terms": set(gt)}


def _sample_terms_per_aspect(per_aspect: int) -> list[str]:
    """Real GO ids grouped by namespace, so cafaeval sees all three aspects with non-zero IA."""
    wanted = ("molecular_function", "biological_process", "cellular_component")
    found: dict[str, list[str]] = {ns: [] for ns in wanted}
    term_id = None
    with open(E.OBO) as handle:
        for line in handle:
            line = line.strip()
            if line == "[Term]":
                term_id = None
            elif line.startswith("id: GO:"):
                term_id = line[4:]
            elif line.startswith("namespace: ") and term_id:
                namespace = line[len("namespace: "):]
                if namespace in found and len(found[namespace]) < per_aspect:
                    found[namespace].append(term_id)
            if all(len(terms) == per_aspect for terms in found.values()):
                break
    return [term for terms in found.values() for term in terms]


def test_coverage_counts_proteins_that_produced_any_go_id():
    records = [
        _record("P1", "<think>reasoning</think> GO:0003674", ["GO:0003674"]),
        _record("P2", "<think>no id at all</think> nothing here", ["GO:0008150"]),
    ]
    diag = E.prediction_diagnostics(records)
    assert diag["n_records"] == 2
    assert diag["n_covered"] == 1
    assert diag["coverage"] == 0.5


def test_recall_ceiling_reflects_ground_truth_sitting_on_uncovered_proteins():
    """cafa_eval's `norm='cafa'` divides recall by every ground-truth protein, not the covered ones."""
    records = [
        _record("P1", "GO:0003674", ["GO:0003674"]),
        # Nine ground-truth terms the model can never be credited for: it emitted no id.
        _record("P2", "no prediction", [f"GO:000{i}" for i in range(1, 10)]),
    ]
    diag = E.prediction_diagnostics(records)
    assert diag["coverage"] == 0.5
    assert diag["recall_ceiling_from_coverage"] == 0.1


def test_think_only_fraction_isolates_terms_that_final_answer_only_would_drop():
    records = [
        _record(
            "P1",
            "<think>maybe GO:0000001 or GO:0000002</think> Final: GO:0000003",
            ["GO:0000003"],
        ),
    ]
    diag = E.prediction_diagnostics(records)
    assert diag["pred_terms_per_protein"] == 3
    # Two of the three predicted ids exist only inside the think block.
    assert diag["think_only_term_fraction"] == 2 / 3
    assert diag["coverage_final_answer_only"] == 1.0


def test_final_answer_only_changes_what_counts_as_a_prediction():
    records = [_record("P1", "<think>GO:0000001</think> no final id", ["GO:0000001"])]
    assert E.prediction_diagnostics(records)["coverage"] == 1.0
    assert E.prediction_diagnostics(records, final_answer_only=True)["coverage"] == 0.0


def test_empty_input_is_zero_not_a_division_error():
    diag = E.prediction_diagnostics([])
    assert diag["coverage"] == 0.0
    assert diag["recall_ceiling_from_coverage"] == 0.0
    assert diag["think_only_term_fraction"] == 0.0


@needs_reference_data
def test_score_generations_keeps_the_metric_of_record_untouched(tmp_path):
    """Diagnostics are additive and opt-in; the sealed path must emit exactly what it always did."""
    records = [_record("P1", "GO:0003674", ["GO:0003674"])]
    plain = E.score_generations(records, str(tmp_path / "plain"))
    assert not any(key.startswith("diag/") for key in plain)

    with_diag = E.score_generations(records, str(tmp_path / "diag"), diagnostics=True)
    for key, value in plain.items():
        assert with_diag[key] == value
    assert with_diag["diag/coverage"] == 1.0


@needs_reference_data
def test_coverage_alone_bounds_weighted_fmax_at_two_c_over_one_plus_c(tmp_path):
    """The bound that makes coverage the first diagnostic to read.

    Under `norm='cafa'`, a protein with no predicted GO id does not enter the precision average but
    still enters the recall denominator. With perfect predictions on the covered share `c`,
    precision is 1 and recall is `c`, so F_max lands on the harmonic mean 2c/(1+c) — 0.5 coverage
    caps the metric at ~0.67 no matter how good the covered predictions are.
    """
    # Real non-root terms, one per aspect. The three ontology roots carry no information accretion,
    # so an IA-weighted score built only from them is 0 and would prove nothing.
    gt = _sample_terms_per_aspect(3)
    assert len(gt) == 9
    records = []
    for i in range(50):
        covered = i < 25  # c = 0.5
        response = "<think>x</think> " + (" ".join(gt) if covered else "no identifier")
        records.append(
            {"protein_id": f"P{i:04d}", "generated_response": response, "gt_terms": set(gt)}
        )

    metrics = E.score_generations(records, str(tmp_path / "half"), diagnostics=True)
    assert metrics["diag/coverage"] == 0.5
    assert metrics["diag/recall_ceiling_from_coverage"] == 0.5
    assert metrics["weighted_fmax"] == pytest.approx(2 * 0.5 / 1.5, abs=0.03)


@needs_reference_data
def test_records_are_consumed_once_even_from_a_generator(tmp_path):
    """score_generations must not exhaust a generator before the diagnostics pass sees it."""
    records = (
        _record(f"P{i}", f"GO:000000{i}", [f"GO:000000{i}"]) for i in range(1, 4)
    )
    metrics = E.score_generations(records, str(tmp_path / "gen"), diagnostics=True)
    assert metrics["diag/n_records"] == 3
    assert metrics["diag/coverage"] == 1.0


def test_in_loop_val_eval_requests_diagnostics_but_the_sealed_path_does_not():
    """Wiring check: _generation_eval needs a GPU, so assert the contract at the source level."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")
    assert 'diagnostics=(split == "val")' in source
    # run_sealed_eval feeds eval.score_generations through eval.evaluate, which defaults to off.
    assert "diagnostics=True" not in source
