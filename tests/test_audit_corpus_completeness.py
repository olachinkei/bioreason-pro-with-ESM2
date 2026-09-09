"""Unit tests for scripts/audit_corpus_completeness.py's pure counting pass (no network)."""

from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from audit_corpus_completeness import summarize_rows  # noqa: E402


def _row(pid, reasoning="", interpro="", ppi="", loc="", mf=None, bp=None, cc=None,
         final_answer="", protein_function=None):
    return {
        "protein_id": pid, "sequence": "MAAA",
        "reasoning": reasoning, "interpro_formatted": interpro, "ppi_formatted": ppi,
        "subcellular_location": loc,
        "go_mf": mf or [], "go_bp": bp or [], "go_cc": cc or [],
        "final_answer": final_answer, "protein_function": protein_function,
    }


def test_counts_reasoning_and_context_and_gt_emptiness():
    rows = [
        _row("P1", reasoning="cites IPR000001", interpro="- IPR000001: x [1-10]",
             mf=["GO:0000001"], bp=[], cc=["GO:0000003"]),
        _row("P2", reasoning="", interpro="", mf=[], bp=["GO:0000002"], cc=[]),
    ]
    summary = summarize_rows(rows)
    assert summary["rows"] == 2
    assert summary["reasoning_nonempty"] == 1
    assert summary["reasoning_nonempty_frac"] == 0.5
    assert summary["context_nonempty"]["interpro_formatted"] == 1
    assert summary["context_nonempty"].get("ppi_formatted", 0) == 0
    # go_mf empty for P2, go_bp empty for P1, go_cc empty for P2
    assert summary["gt_empty"]["go_mf"] == 1
    assert summary["gt_empty"]["go_bp"] == 1
    assert summary["gt_empty"]["go_cc"] == 1


def test_skips_rows_missing_id_or_sequence():
    rows = [
        {"protein_id": "", "sequence": "MAAA"},
        {"protein_id": "P1", "sequence": ""},
        _row("P2"),
    ]
    summary = summarize_rows(rows)
    assert summary["rows"] == 1


def test_has_reasoning_column_reflects_schema_not_just_emptiness():
    rows_without_column = [
        {"protein_id": "P1", "sequence": "MAAA", "go_mf": [], "go_bp": [], "go_cc": []},
    ]
    summary = summarize_rows(rows_without_column)
    assert summary["has_reasoning_column"] is False
    assert summary["reasoning_nonempty"] == 0

    rows_with_empty_column = [_row("P1", reasoning="")]
    summary2 = summarize_rows(rows_with_empty_column)
    assert summary2["has_reasoning_column"] is True
    assert summary2["reasoning_nonempty"] == 0


def test_empty_corpus_does_not_divide_by_zero():
    summary = summarize_rows([])
    assert summary["rows"] == 0
    assert summary["reasoning_nonempty_frac"] == 0.0
    assert summary["gt_empty_frac"]["go_mf"] == 0.0


def test_counts_final_answer_and_known_protein_function():
    rows = [
        _row("P1", final_answer="- Functional Summary: x", protein_function="Binds calcium."),
        _row("P2", final_answer="- Functional Summary: y", protein_function="Not known"),
        _row("P3", final_answer="", protein_function=None),
    ]
    summary = summarize_rows(rows)
    assert summary["has_final_answer_column"] is True
    assert summary["final_answer_nonempty"] == 2
    assert summary["final_answer_nonempty_frac"] == 2 / 3
    assert summary["has_protein_function_column"] is True
    assert summary["protein_function_known"] == 1  # P2's "Not known" doesn't count, P3's None doesn't
    assert summary["protein_function_known_frac"] == 1 / 3


def test_has_final_answer_column_reflects_schema_not_just_emptiness():
    rows_without_column = [
        {"protein_id": "P1", "sequence": "MAAA", "go_mf": [], "go_bp": [], "go_cc": []},
    ]
    summary = summarize_rows(rows_without_column)
    assert summary["has_final_answer_column"] is False
    assert summary["has_protein_function_column"] is False
