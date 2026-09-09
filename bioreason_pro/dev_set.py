"""bioreason_pro.dev_set — pure helpers for the Phase 1 temporal development set (ADR-022/024).

Building the actual `data/cafa_no_knowledge_dev_set.jsonl` needs network access (UniProt, optionally
EBI InterProScan) and lives in `scripts/build_cafa_no_knowledge_dev_set.py`. The logic that decides
*which* protein ids belong in the dev set, and that asserts it is disjoint from the sealed holdout and
the training corpus, is pure and lives here so it can be unit-tested without either network call —
method rule 1 (R1 in plan.md): disjointness must be asserted in code, not eyeballed.
"""

from __future__ import annotations

# CAFA's evaluation-namespace codes -> this project's aspect columns. Matches
# bioreason_pro.cafa_fmax.NS_TO_ASPECT (cafaeval's own labels), because both describe the same three
# GO namespaces; kept as a separate mapping here since this one keys off the raw CAFA TSV column,
# not cafaeval's output.
ASPECT_TO_COLUMN = {"MFO": "go_mf", "BPO": "go_bp", "CCO": "go_cc"}


class DevSetError(AssertionError):
    """Raised when the temporal dev set fails a disjointness or size invariant."""


# Found 2026-08-21 while building this dev set (plan.md Phase 1, gate R1): streaming the full
# training corpus (wanglab/bioreason-pro-sft-reasoning-data, 117,002 unique proteins, +
# wanglab/bioreason-pro-rl-reasoning-data, 9,154) turned up 8 proteins that are CAFA-no-knowledge at
# t0 (zero rows in known_t0.tsv — CAFA's own file agrees they were unannotated) yet are also present
# in the training corpus. This does not contradict ADR-023/024's own "~0.3% residue" observation about
# the corpus; it is a concrete instance of it landing inside this specific 1,717-protein universe.
# Excluded for the same reason the 213 sealed-holdout overlaps are excluded: the non-negotiable
# train/val/test disjointness constraint means a "held-out" dev set cannot contain proteins the model
# trained on. Owner-confirmed 2026-08-21. Re-verified by scripts/verify_cafa_no_knowledge_disjoint_from_corpus.py.
KNOWN_TRAINING_CORPUS_OVERLAP_IDS: frozenset[str] = frozenset({
    "F1LQY6", "P0AE39", "P76115", "Q6PEH5", "Q9P6I4",  # wanglab/bioreason-pro-sft-reasoning-data
    "P41822", "Q59PG6", "Q78T81",                       # wanglab/bioreason-pro-rl-reasoning-data
})


def parse_no_knowledge_terms(rows) -> dict[str, dict[str, set[str]]]:
    """`rows`: iterable of {"EntryID", "term", "aspect"} dicts (one CAFA tsv row each).

    Returns {protein_id: {"go_mf": {...}, "go_bp": {...}, "go_cc": {...}}}, aspects present only when
    at least one term was seen for that protein (mirrors how the wanglab schema leaves an aspect
    column absent/None rather than an empty list — see eval_targets.base.parse_gt).
    """
    out: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        pid = row["EntryID"]
        aspect = row["aspect"]
        column = ASPECT_TO_COLUMN.get(aspect)
        if column is None:
            raise DevSetError(f"{pid}: unknown CAFA aspect code {aspect!r}, expected one of {sorted(ASPECT_TO_COLUMN)}")
        out.setdefault(pid, {}).setdefault(column, set()).add(row["term"])
    return out


def build_dev_ids(
    no_knowledge_ids: set[str],
    holdout_ids: set[str],
    *,
    corpus_overlap_ids: frozenset[str] = KNOWN_TRAINING_CORPUS_OVERLAP_IDS,
    expected_count: int = 1496,
) -> set[str]:
    """CAFA's no-knowledge targets, minus the pinned sealed holdout and the known training-corpus
    overlap — plan.md Phase 1's dev set.

    The ~213 ids CAFA's no-knowledge list shares with the sealed holdout, and the 8 it shares with the
    training corpus, are *expected* and are what this function removes — that overlap is not itself an
    error. What fails closed is anything left over afterwards: a residual holdout or corpus overlap in
    the returned set, or a count that doesn't match what was verified against the full holdout/corpus
    id lists, either of which would mean checkpoint-selection could quietly touch sealed-test or
    trained-on proteins (ADR-002, and the non-negotiable train/val/test disjointness constraint).
    """
    dev_ids = no_knowledge_ids - holdout_ids - corpus_overlap_ids
    residual_holdout = dev_ids & holdout_ids
    residual_corpus = dev_ids & corpus_overlap_ids
    if residual_holdout or residual_corpus:  # impossible by construction of set difference; defensive
        raise DevSetError(
            f"{len(residual_holdout)} dev-set ids are still in the sealed holdout, "
            f"{len(residual_corpus)} still overlap the training corpus"
        )
    if len(dev_ids) != expected_count:
        removed_holdout = len(no_knowledge_ids & holdout_ids)
        removed_corpus = len((no_knowledge_ids - holdout_ids) & corpus_overlap_ids)
        raise DevSetError(
            f"expected {expected_count} dev-set ids, got {len(dev_ids)} "
            f"({len(no_knowledge_ids)} no-knowledge ids, {removed_holdout} removed as holdout overlap, "
            f"{removed_corpus} removed as training-corpus overlap)"
        )
    return dev_ids


def assert_disjoint_from_corpus(dev_ids: set[str], corpus_ids: set[str]) -> None:
    """Assert the dev set shares no protein with the training corpus (either SFT or RL reasoning data).

    Not a formality: this is exactly the check that found ADR-029's 8-protein overlap. CAFA's
    no-knowledge targets having no experimental annotation as of t0 turned out not to structurally
    guarantee absence from the training corpus, which method rule 6/12 already warns against assuming
    without checking — call this after excluding `KNOWN_TRAINING_CORPUS_OVERLAP_IDS` (`build_dev_ids`
    does, by default) so a clean run here confirms that exclusion list is still complete, not that the
    corpus and CAFA's list happen to be disjoint on their own.
    """
    overlap = dev_ids & corpus_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise DevSetError(
            f"{len(overlap)} dev-set ids overlap the training corpus ({sample}, ...); "
            "the temporal dev set must be disjoint from every protein the model trains on"
        )
