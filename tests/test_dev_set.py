"""Unit tests for bioreason_pro.dev_set — plan.md Phase 1's temporal dev-set construction (pure, no
network). Covers the disjointness assertions method rule 1/R1 requires to be in code, not eyeballed.
"""

import pytest

from bioreason_pro.dev_set import (
    KNOWN_TRAINING_CORPUS_OVERLAP_IDS,
    DevSetError,
    assert_disjoint_from_corpus,
    build_dev_ids,
    parse_no_knowledge_terms,
)


def _row(pid, term, aspect):
    return {"EntryID": pid, "term": term, "aspect": aspect}


def test_parse_no_knowledge_terms_groups_by_protein_and_aspect():
    rows = [
        _row("P1", "GO:0003674", "MFO"),
        _row("P1", "GO:0005488", "MFO"),
        _row("P1", "GO:0009987", "BPO"),
        _row("P2", "GO:0005575", "CCO"),
    ]
    parsed = parse_no_knowledge_terms(rows)
    assert parsed == {
        "P1": {"go_mf": {"GO:0003674", "GO:0005488"}, "go_bp": {"GO:0009987"}},
        "P2": {"go_cc": {"GO:0005575"}},
    }


def test_parse_no_knowledge_terms_rejects_unknown_aspect():
    with pytest.raises(DevSetError, match="unknown CAFA aspect"):
        parse_no_knowledge_terms([_row("P1", "GO:0003674", "XXX")])


def test_build_dev_ids_subtracts_holdout():
    no_knowledge = {"A", "B", "C"}
    holdout = {"B"}
    assert build_dev_ids(no_knowledge, holdout, expected_count=2) == {"A", "C"}


def test_build_dev_ids_matches_plan_scale_before_corpus_exclusion():
    """Holdout-only subtraction reaches the plan's stated 1,717 -> 1,504 (ADR-024)."""
    no_knowledge = {f"P{i}" for i in range(1717)}
    holdout = {f"P{i}" for i in range(213)}
    dev = build_dev_ids(no_knowledge, holdout, corpus_overlap_ids=frozenset(), expected_count=1504)
    assert len(dev) == 1504
    assert dev.isdisjoint(holdout)


def test_build_dev_ids_excludes_known_training_corpus_overlap_by_default():
    """1,717 - 213 holdout - 8 corpus-overlap = 1,496 (the corrected Phase 1 dev-set size)."""
    no_knowledge = {f"P{i}" for i in range(1717 - len(KNOWN_TRAINING_CORPUS_OVERLAP_IDS))}
    no_knowledge |= set(KNOWN_TRAINING_CORPUS_OVERLAP_IDS)
    holdout = {f"P{i}" for i in range(213)}
    dev = build_dev_ids(no_knowledge, holdout, expected_count=1496)
    assert len(dev) == 1496
    assert dev.isdisjoint(KNOWN_TRAINING_CORPUS_OVERLAP_IDS)
    assert dev.isdisjoint(holdout)


def test_build_dev_ids_wrong_expected_count_raises():
    with pytest.raises(DevSetError, match="expected 999"):
        build_dev_ids({"A", "B"}, set(), corpus_overlap_ids=frozenset(), expected_count=999)


def test_assert_disjoint_from_corpus_passes_when_disjoint():
    assert_disjoint_from_corpus({"A", "B"}, {"C", "D"})  # must not raise


def test_assert_disjoint_from_corpus_raises_on_overlap():
    with pytest.raises(DevSetError, match="overlap the training corpus"):
        assert_disjoint_from_corpus({"A", "B"}, {"B", "C"})
