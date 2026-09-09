"""Supervision-target variants.

`full_closure` must stay byte-identical to what the repository has always trained on; `leaf_only`
exists because the shipped label columns are ancestor-closed, so 86% of the supervision is terms the
scorer re-derives by propagation.
"""

from __future__ import annotations

import pytest

from bioreason_pro import data_contract as DC

# GO:0000003 -> GO:0000002 -> GO:0000001 within one aspect, plus an unrelated leaf.
ANCESTORS = {
    "GO:0000003": {"GO:0000002", "GO:0000001"},
    "GO:0000002": {"GO:0000001"},
    "GO:0000001": set(),
    "GO:0000009": set(),
}

ROW = {
    "protein_id": "P1",
    "sequence": "MKT",
    "go_mf": ["GO:0000001", "GO:0000002", "GO:0000003"],
    "go_bp": ["GO:0000009"],
    "go_cc": [],
}


@pytest.fixture(autouse=True)
def _stub_ancestors(monkeypatch):
    """Avoid parsing the 31 MB ontology in unit tests."""
    monkeypatch.setattr(DC, "_go_ancestors_for_targets", lambda: ANCESTORS)


def test_default_variant_is_the_shipped_full_closure(monkeypatch):
    monkeypatch.delenv("SENPAI_TARGET_VARIANT", raising=False)
    assert DC.active_target_variant() == "full_closure"
    assert DC.format_go_answer(ROW) == (
        "MF: GO:0000001, GO:0000002, GO:0000003\nBP: GO:0000009"
    )


def test_leaf_only_keeps_the_most_specific_term_per_aspect(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    assert DC.format_go_answer(ROW) == "MF: GO:0000003\nBP: GO:0000009"


def test_leaf_only_preserves_the_aspect_grouping_and_ordering(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    row = {**ROW, "go_cc": ["GO:0000009", "GO:0000003"]}
    answer = DC.format_go_answer(row)
    assert answer.splitlines()[0].startswith("MF: ")
    assert answer.splitlines()[-1].startswith("CC: ")
    # Terms stay sorted so the target is deterministic across runs.
    assert "CC: GO:0000003, GO:0000009" in answer


def test_explicit_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    assert DC.format_go_answer(ROW, variant="full_closure") == (
        "MF: GO:0000001, GO:0000002, GO:0000003\nBP: GO:0000009"
    )


def test_unknown_variant_fails_closed(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "shorter_please")
    with pytest.raises(ValueError, match="not a known target variant"):
        DC.active_target_variant()
    with pytest.raises(ValueError, match="unknown target variant"):
        DC.format_go_answer(ROW, variant="shorter_please")


def test_a_row_with_no_annotations_is_unchanged_by_the_variant(monkeypatch):
    empty = {"protein_id": "P1", "sequence": "MKT", "go_mf": [], "go_bp": [], "go_cc": []}
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    assert DC.format_go_answer(empty) == "No approved GO annotations."


def test_leaf_only_is_recoverable_by_propagation(monkeypatch):
    """The variant may shrink the target only in ways propagation puts back."""
    from bioreason_pro.rewards import propagate_ancestors

    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    leaf_answer = DC.format_go_answer(ROW)
    full_answer = DC.format_go_answer(ROW, variant="full_closure")

    leaf_terms = set(DC.parse_go_terms(leaf_answer))
    full_terms = set(DC.parse_go_terms(full_answer))
    assert leaf_terms < full_terms
    assert propagate_ancestors(leaf_terms, ANCESTORS) == propagate_ancestors(full_terms, ANCESTORS)


def test_training_runs_record_which_variant_they_used():
    """A condition search whose runs do not record their condition is not auditable."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")
    assert '"target_variant": active_target_variant()' in source
    assert '"prompt_variant": active_prompt_variant()' in source


def test_leaf_mf_only_strips_mf_and_leaves_bp_cc_closed(monkeypatch):
    """Job 627: leaf targets lifted MF 99.5% but cost BP 22.9% and CC 42.4%."""
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_mf_only")
    row = {
        "protein_id": "P1",
        "sequence": "MKT",
        "go_mf": ["GO:0000001", "GO:0000002", "GO:0000003"],
        "go_bp": ["GO:0000001", "GO:0000002", "GO:0000003"],
        "go_cc": ["GO:0000009"],
    }
    answer = DC.format_go_answer(row)
    assert "MF: GO:0000003" in answer
    # BP keeps the shipped closure so the model can still hedge at a mid-level term.
    assert "BP: GO:0000001, GO:0000002, GO:0000003" in answer
    assert "CC: GO:0000009" in answer


def test_every_variant_declares_which_aspects_it_strips():
    assert set(DC.LEAF_ASPECTS) == set(DC.TARGET_VARIANTS)
    assert DC.LEAF_ASPECTS["full_closure"] == frozenset()
    assert DC.LEAF_ASPECTS["leaf_only"] == frozenset({"MF", "BP", "CC"})
    assert DC.LEAF_ASPECTS["leaf_mf_only"] == frozenset({"MF"})
