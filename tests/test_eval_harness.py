"""Units for the eval harness (eval.py).

Pure TSV-writing tests always run. The end-to-end test runs the REAL cafaeval against the real
go-basic.obo/IA.txt (gitignored → skipped in CI) to validate the full extract→TSV→score path.
"""

from pathlib import Path

import pytest

import eval as E


@pytest.mark.parametrize(
    ("split", "requested", "expected"),
    [("val", None, 256), ("val", 3, 3), ("test", None, None), ("test", 3, 3)],
)
def test_cli_subset_contract(split, requested, expected):
    assert E._resolve_cli_subset_size(split, requested) == expected


def test_cli_subset_rejects_non_positive_value():
    with pytest.raises(ValueError, match="positive"):
        E._resolve_cli_subset_size("test", 0)


def test_write_cafa_predictions_scores_and_sets(tmp_path):
    d = E.write_cafa_predictions(
        [("P1", {"GO:0005515": 0.9, "GO:0009987": 0.5}), ("P2", {"GO:0005737"})],
        str(tmp_path / "preds"),
    )
    body = Path(d, "predictions.tsv").read_text().splitlines()
    assert "P1\tGO:0005515\t0.9" in body
    assert "P2\tGO:0005737\t1.0" in body            # set → default score 1.0
    assert len(body) == 3


def test_write_cafa_ground_truth(tmp_path):
    gt = tmp_path / "gt.tsv"
    E.write_cafa_ground_truth([("P1", {"GO:0005515"}), ("P2", {"GO:0005737"})], str(gt))
    lines = set(gt.read_text().splitlines())
    assert lines == {"P1\tGO:0005515", "P2\tGO:0005737"}


REAL_OBO = Path(E.OBO)
REAL_IA = Path(E.IA)
try:
    from bioreason_pro.license_policy import validate_reference_assets

    validate_reference_assets(use="evaluation")
    _HAVE_DATA = True
except Exception:
    _HAVE_DATA = False


@pytest.mark.skipif(not _HAVE_DATA, reason="approved go-basic.obo + IA.txt are not installed")
def test_score_generations_end_to_end(tmp_path):
    perfect = [
        {"protein_id": "P1", "generated_response": "<think>x</think> GO:0005515 GO:0009987",
         "gt_terms": {"GO:0005515", "GO:0009987"}},
        {"protein_id": "P2", "generated_response": "ans GO:0005737", "gt_terms": {"GO:0005737"}},
    ]
    m = E.score_generations(perfect, str(tmp_path / "perfect"))
    assert set(m) >= {"weighted_fmax", "weighted_fmax_mf", "weighted_fmax_bp", "weighted_fmax_cc"}
    assert m["weighted_fmax"] == pytest.approx(1.0)

    # A WRONG term (false positive) lowers precision. (A merely-missing term does not, since CAFA
    # F_max precision is averaged only over predicted proteins — coverage-based.)
    degraded = [
        {"protein_id": "P1", "generated_response": "GO:0016787",   # hydrolase activity — wrong MF
         "gt_terms": {"GO:0005515"}},                              # ground truth: protein binding
        {"protein_id": "P2", "generated_response": "GO:0005737", "gt_terms": {"GO:0005737"}},
    ]
    m2 = E.score_generations(degraded, str(tmp_path / "degraded"))
    assert 0.0 <= m2["weighted_fmax_mf"] < 1.0     # wrong MF term lowers the MF score
    assert m2["weighted_fmax"] < 1.0
