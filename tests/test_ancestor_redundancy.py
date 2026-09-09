"""The scored prediction set is invariant to explicitly-named ancestors.

Measured on the Phase 4 GRPO rollouts, 65.2% of every completion's GO ids were ancestors of another
id in the same completion. Removing them left weighted F_max unchanged to six decimal places, so
that share of a fixed generation budget buys nothing. These tests pin the invariant behind that
claim: the pure set operation, and the end-to-end score through the real cafaeval.
"""

from __future__ import annotations

import pytest

import eval as E
from bioreason_pro.go_obo import load_go_ancestors
from bioreason_pro.rewards import propagate_ancestors, strip_implied_ancestors

try:
    from bioreason_pro.license_policy import validate_reference_assets

    validate_reference_assets(use="evaluation")
    _HAVE_DATA = True
except Exception:
    _HAVE_DATA = False

needs_reference_data = pytest.mark.skipif(
    not _HAVE_DATA, reason="approved go-basic.obo + IA.txt are not installed"
)

# A -> B -> C, plus an unrelated D.
ANCESTORS = {
    "GO:0000003": {"GO:0000002", "GO:0000001"},
    "GO:0000002": {"GO:0000001"},
    "GO:0000001": set(),
    "GO:0000009": set(),
}


def test_stripping_keeps_only_the_most_specific_terms():
    terms = {"GO:0000001", "GO:0000002", "GO:0000003", "GO:0000009"}
    assert strip_implied_ancestors(terms, ANCESTORS) == {"GO:0000003", "GO:0000009"}


def test_stripping_is_a_no_op_when_nothing_is_implied():
    terms = {"GO:0000003", "GO:0000009"}
    assert strip_implied_ancestors(terms, ANCESTORS) == terms


def test_stripping_then_propagating_reproduces_the_original_closure():
    """The round trip is what makes the explicit ancestors redundant rather than merely small."""
    terms = {"GO:0000001", "GO:0000002", "GO:0000003", "GO:0000009"}
    stripped = strip_implied_ancestors(terms, ANCESTORS)
    assert propagate_ancestors(stripped, ANCESTORS) == propagate_ancestors(terms, ANCESTORS)


@needs_reference_data
def test_real_cafaeval_scores_a_stripped_prediction_identically(tmp_path):
    """End to end: the metric of record cannot tell the two prediction sets apart."""
    ancestors = load_go_ancestors(E.OBO)
    # Deep, real terms so each one drags a chain of ancestors along with it.
    leaves = ["GO:0000902", "GO:0001501", "GO:0006950", "GO:0030855"]
    full = set(leaves)
    for leaf in leaves:
        full |= ancestors.get(leaf, set())
    stripped = strip_implied_ancestors(full, ancestors)
    assert len(stripped) < len(full), "the fixture must actually contain implied ancestors"

    def records(terms):
        return [
            {
                "protein_id": f"P{i:03d}",
                "generated_response": " ".join(sorted(terms)),
                "gt_terms": set(leaves),
            }
            for i in range(10)
        ]

    verbose = E.score_generations(records(full), str(tmp_path / "verbose"))
    concise = E.score_generations(records(stripped), str(tmp_path / "concise"))
    assert concise == verbose
