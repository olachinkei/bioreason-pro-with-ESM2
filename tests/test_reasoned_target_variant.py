"""ADR-017: reasoning supervision, and the coupling that keeps it honest.

The dataset ships `reasoning` traces, but sampled 20/20 of them cite InterPro domain ids with
residue ranges and STRING interaction partners. Supervising those against the shipped sequence-only
prompt would train the model to invent `IPR051630 (residues 9-1410)` from an amino acid string:
specific, confident, unknowable, and indistinguishable from a real finding to a reader in drug
discovery. These tests pin that the variant cannot be configured into that state.
"""

from __future__ import annotations

import pytest

from bioreason_pro import data_contract as DC

ROW = {
    "protein_id": "P12345",
    "sequence": "MKTFF",
    "go_mf": ["GO:0042803"],
    "go_bp": ["GO:0000002"],
    "go_cc": ["GO:0005739"],
    "interpro_formatted": "IPR051630 Transcriptional Corepressor (residues 9-1410)",
    "ppi_formatted": "Interacts with: PARTNER1, PARTNER2",
    "subcellular_location": "Nucleus",
    "reasoning": "The polypeptide is dominated by IPR051630, which establishes membership in the "
                 "lysine-specific demethylase lineage.",
    "final_answer": "- Functional Summary: A DNA-binding transcriptional regulator.",
}


def _adapt(row, variant, monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", variant)
    DC._go_ancestors_for_targets.cache_clear()
    return DC.adapt_synthetic_fixture_row(dict(row))


def test_the_shipped_variants_are_untouched(monkeypatch):
    """No context, no reasoning, no summary — byte-identical to what every SFT in plan.md trained on."""
    for variant in ("full_closure", "leaf_only", "leaf_mf_only"):
        out = _adapt(ROW, variant, monkeypatch)
        assert out["prompt"]["assistant_reasoning"] == ""
        assert "IPR051630" not in out["prompt"]["user"]
        assert "PARTNER1" not in out["prompt"]["user"]
        # ROW carries final_answer, but the summary splice (Phase 4) is reasoned-variant-only
        assert "UniProt Summary" not in out["prompt"]["assistant_answer"]
        assert out["prompt"]["assistant_answer"] == DC.format_go_answer(ROW, variant=variant)
        assert "functional summary" not in out["prompt"]["user"].lower()


def test_the_reasoned_variant_supervises_the_trace_and_shows_it_the_evidence(monkeypatch):
    out = _adapt(ROW, "leaf_only_reasoned", monkeypatch)
    assert out["prompt"]["assistant_reasoning"].startswith("The polypeptide is dominated")
    user = out["prompt"]["user"]
    # the trace cites IPR051630 and interaction partners, so the prompt must contain them
    assert "IPR051630" in user
    assert "PARTNER1" in user
    assert "Nucleus" in user
    # Phase 4: the functional summary now leads the answer, GO lines follow
    assert out["prompt"]["assistant_answer"].startswith("- Functional Summary:")
    assert "MF:" in out["prompt"]["assistant_answer"]


def test_reasoning_without_its_evidence_is_refused(monkeypatch):
    """The failure this variant exists to prevent: supervising a trace the prompt cannot support."""
    stripped = {k: v for k, v in ROW.items()
                if k not in ("interpro_formatted", "ppi_formatted", "subcellular_location")}
    with pytest.raises(ValueError, match="fabricate"):
        _adapt(stripped, "leaf_only_reasoned", monkeypatch)


def test_an_empty_trace_is_refused_rather_than_silently_reintroducing_adr_015(monkeypatch):
    blank = {**ROW, "reasoning": "   "}
    with pytest.raises(ValueError, match="requires a non-empty `reasoning`"):
        _adapt(blank, "leaf_only_reasoned", monkeypatch)


def test_the_go_answer_is_still_authored_from_go_labels_the_summary_only_leads_it(monkeypatch):
    """Phase 4 / gate R3: `final_answer` is now adopted, spliced ahead of the GO lines — but the
    GO lines themselves stay authored from the label columns, not from prose. Adopting the summary
    must not let prose stand in for the scored GO answer."""
    out = _adapt({**ROW, "final_answer": "A nuclear histone demethylase that ..."},
                 "leaf_only_reasoned", monkeypatch)
    assert "nuclear histone demethylase" in out["prompt"]["assistant_answer"]  # the summary, adopted
    assert "GO:" in out["prompt"]["assistant_answer"]  # the GO lines, still authored from labels
    assert out["prompt"]["assistant_answer"].endswith(DC.format_go_answer(ROW))


def test_missing_final_answer_is_refused_for_sft(monkeypatch):
    """SFT now supervises the spliced summary, so its target must actually be present — the same
    fail-closed treatment as a missing `reasoning` trace, for the same reason (ADR-015 lineage)."""
    no_summary = {k: v for k, v in ROW.items() if k != "final_answer"}
    with pytest.raises(ValueError, match="functional"):
        _adapt(no_summary, "leaf_only_reasoned", monkeypatch)


def test_rl_and_eval_do_not_require_final_answer_either(monkeypatch):
    """Only SFT supervises the spliced summary; RL scores rollouts and eval has no target at all,
    so neither should be blocked by a missing `final_answer` any more than by missing `reasoning`."""
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    no_summary = {k: v for k, v in ROW.items() if k != "final_answer"}
    for use in ("rl-training", "holdout-evaluation"):
        out = DC.adapt_synthetic_fixture_row(dict(no_summary), use=use)
        assert "UniProt Summary" not in out["prompt"]["assistant_answer"]
        assert out["prompt"]["assistant_answer"].startswith(("MF:", "BP:", "CC:"))


def test_compose_functional_summary_splices_uniprot_text_after_the_first_line():
    """Mirrors upstream's `_add_uniprot_summary` (bowang-lab/BioReason-Pro, dataset/cafa5/load.py):
    the header line stays first, the UniProt text becomes its own second line, the rest follows."""
    composed = DC.compose_functional_summary(
        "- Functional Summary: A pore-forming outer membrane protein.\n- Interaction Partners: none",
        "Forms pores that allow passive diffusion of small molecules.",
    )
    lines = composed.split("\n")
    assert lines[0] == "- Functional Summary: A pore-forming outer membrane protein."
    assert lines[1] == "- UniProt Summary: Forms pores that allow passive diffusion of small molecules."
    assert lines[2] == "- Interaction Partners: none"


def test_compose_functional_summary_handles_a_missing_protein_function():
    composed = DC.compose_functional_summary("- Functional Summary: Unknown protein.", None)
    assert composed == "- Functional Summary: Unknown protein.\n- UniProt Summary: "


def test_functional_summary_instruction_asks_for_one_only_when_known():
    known = DC.functional_summary_instruction("Mediates potassium-chloride cotransport.")
    assert "summary" in known.lower()
    assert "not known" not in known.lower()

    for unknown in ("Not known", "not known", "  ", None):
        instruction = DC.functional_summary_instruction(unknown)
        assert "not known" in instruction.lower()
        assert "do not attempt" in instruction.lower()


def test_the_reasoned_prompt_carries_the_conditional_summary_instruction(monkeypatch):
    known = _adapt({**ROW, "protein_function": "Mediates potassium-chloride cotransport."},
                    "leaf_only_reasoned", monkeypatch)
    assert "functional summary" in known["prompt"]["user"].lower()
    assert "not known" not in known["prompt"]["user"].lower()

    unknown = _adapt({**ROW, "protein_function": "Not known"}, "leaf_only_reasoned", monkeypatch)
    assert "not known" in unknown["prompt"]["user"].lower()


# --- organism (plan.md Phase 5) -------------------------------------------------------------------

def test_organism_renders_when_present(monkeypatch):
    out = _adapt({**ROW, "organism": "Rattus norvegicus (Rat)"}, "leaf_only_reasoned", monkeypatch)
    assert "Organism (UniProtKB): Rattus norvegicus (Rat)" in out["prompt"]["user"]


def test_organism_renders_as_not_available_when_missing(monkeypatch):
    out = _adapt(ROW, "leaf_only_reasoned", monkeypatch)  # ROW carries no organism key
    assert "Organism (UniProtKB): not available" in out["prompt"]["user"]


def test_organism_is_not_shipped_to_non_reasoned_variants(monkeypatch):
    for variant in ("full_closure", "leaf_only", "leaf_mf_only"):
        out = _adapt({**ROW, "organism": "Rattus norvegicus (Rat)"}, variant, monkeypatch)
        assert "Organism" not in out["prompt"]["user"]
        assert "Rattus norvegicus" not in out["prompt"]["user"]


def test_organism_does_not_gate_the_sft_reasoning_evidence_requirement(monkeypatch):
    """Unlike CONTEXT_COLUMNS, a missing `organism` must never raise: a 60-trace spot check found no
    reasoning trace names its own organism, so there is no fabrication risk to guard against, and this
    field must not become a second, undocumented way for an otherwise-fine row to be skipped."""
    no_organism = {k: v for k, v in ROW.items() if k != "organism"}
    out = _adapt(no_organism, "leaf_only_reasoned", monkeypatch)  # must not raise
    assert out["prompt"]["assistant_reasoning"]

    # the reverse also holds: organism present does not substitute for the real evidence requirement
    stripped = {k: v for k, v in ROW.items()
                if k not in ("interpro_formatted", "ppi_formatted", "subcellular_location")}
    with pytest.raises(ValueError, match="fabricate"):
        _adapt({**stripped, "organism": "Rattus norvegicus (Rat)"}, "leaf_only_reasoned", monkeypatch)


# --- GO-GPT predictions (ADR-037) ------------------------------------------------------------------
#
# Unlike organism, a 60-trace spot check found 56/60 reasoning traces cite a GO id verbatim from
# go_pred — real evidence a trace builds on, not a supplementary signal. So this column is folded
# into CONTEXT_COLUMNS (fabrication-gated), not rendered like ORGANISM_COLUMN.

def test_gogpt_prediction_renders_in_prompt_for_reasoned_variant(monkeypatch):
    out = _adapt({**ROW, "go_pred": "Molecular Function (MF): GO:0042803 (protein homodimerization "
                                     "activity)"}, "leaf_only_reasoned", monkeypatch)
    assert "Predicted GO terms (GO-GPT):\nMolecular Function (MF): GO:0042803" in out["prompt"]["user"]


def test_gogpt_prediction_renders_as_not_available_when_missing(monkeypatch):
    """ROW's own interpro/ppi/subcellular satisfy the evidence gate on their own, so a row simply
    missing go_pred (as opposed to missing everything) must render the placeholder, not raise."""
    out = _adapt(ROW, "leaf_only_reasoned", monkeypatch)  # ROW carries no go_pred key
    assert "Predicted GO terms (GO-GPT):\nnot available" in out["prompt"]["user"]


def test_gogpt_prediction_is_not_shipped_to_non_reasoned_variants(monkeypatch):
    for variant in ("full_closure", "leaf_only", "leaf_mf_only"):
        out = _adapt({**ROW, "go_pred": "Molecular Function (MF): GO:0042803 (...)"},
                     variant, monkeypatch)
        assert "Predicted GO terms" not in out["prompt"]["user"]
        assert "GO:0042803" not in out["prompt"]["user"]


def test_gogpt_prediction_alone_gates_sft_evidence_closed_when_it_is_the_only_source_missing(monkeypatch):
    """The reverse of the fixture case above: strip everything BUT go_pred, confirm go_pred alone
    satisfies the evidence gate (CONTEXT_COLUMNS uses `any`, not `all` — consistent with the
    pre-existing interpro/ppi/subcellular design, not a new relaxation introduced for this column)."""
    stripped = {k: v for k, v in ROW.items()
                if k not in ("interpro_formatted", "ppi_formatted", "subcellular_location")}
    out = _adapt({**stripped, "go_pred": "Molecular Function (MF): GO:0042803 (...)"},
                 "leaf_only_reasoned", monkeypatch)  # must not raise
    assert out["prompt"]["assistant_reasoning"]


def test_the_coupling_is_declared_as_data(monkeypatch):
    assert "leaf_only_reasoned" in DC.TARGET_VARIANTS
    assert "leaf_only_reasoned" in DC.REASONED_TARGET_VARIANTS
    assert DC.LEAF_ASPECTS["leaf_only_reasoned"] == frozenset({"MF", "BP", "CC"})
    assert [c for c, _ in DC.CONTEXT_COLUMNS] == [
        "interpro_formatted", "ppi_formatted", "subcellular_location", "go_pred"]
    assert DC.ORGANISM_COLUMN == ("organism", "Organism (UniProtKB)")


def test_unknown_variant_still_fails_closed(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_reasoned")  # near-miss
    with pytest.raises(ValueError):
        DC.active_target_variant()


# --------------------------------------------------------------- require_active_variant_matches
#
# plan.md ADR-031: a Phase 3 GPU sweep evaluated leaf_only_reasoned checkpoints with
# SENPAI_TARGET_VARIANT left unset, which silently defaults to full_closure -- a prompt with NO
# InterPro/PPI/subcellular-location context at all. The model still reasoned (that behaviour is
# trained in) but fabricated domain claims instead of leaving them out, since ADR-017's coupling
# between reasoning supervision and evidence-in-prompt only holds when the evidence is actually
# there. These tests pin the guard that catches this before it silently produces a meaningless run.

def test_matching_variant_is_a_noop(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC.require_active_variant_matches("leaf_only_reasoned")  # must not raise


def test_mismatched_variant_fails_closed(monkeypatch):
    monkeypatch.delenv("SENPAI_TARGET_VARIANT", raising=False)  # defaults to full_closure
    with pytest.raises(ValueError, match="does not match"):
        DC.require_active_variant_matches("leaf_only_reasoned")


def test_unrecorded_expected_variant_skips_the_check(monkeypatch):
    """An older checkpoint whose producing run never logged target_variant: don't block on
    missing historical metadata, just don't guarantee anything either."""
    monkeypatch.delenv("SENPAI_TARGET_VARIANT", raising=False)
    DC.require_active_variant_matches(None)  # must not raise


def test_every_fixture_row_can_drive_the_reasoned_variant(monkeypatch):
    """Smoke 966 died on fixture row 1: `reasoning` was None on five of six rows.

    The unit tests all used a hand-built row, so nothing exercised the fixture the GPU smoke
    actually consumes. A fixture that only works for row 0 is a smoke that fails at minute five of
    a scheduled job instead of in 0.1s here.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    rows = [json.loads(line) for line in
            (root / "tests" / "fixtures" / "approved_training_rows.jsonl").read_text().splitlines()
            if line.strip()]
    assert rows, "fixture must not be empty"
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    for i, row in enumerate(rows):
        out = DC.adapt_synthetic_fixture_row(dict(row))
        assert out["prompt"]["assistant_reasoning"], f"row {i} produced an empty trace"


# --- stage awareness (ADR-017 revision after job 970) -------------------------------------------

def test_rl_and_eval_do_not_require_a_reasoning_column(monkeypatch):
    """The RL corpus ships NO `reasoning` at all (100% empty when sampled).

    Only SFT supervises a trace, so only SFT needs one. Requiring it everywhere made the reasoned
    variant unusable for the RL stage and for the sealed evaluation.
    """
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    no_trace = {k: v for k, v in ROW.items() if k != "reasoning"}
    for use in ("rl-training", "holdout-evaluation"):
        out = DC.adapt_synthetic_fixture_row(dict(no_trace), use=use)
        assert out["prompt"]["assistant_reasoning"] == ""
        assert "IPR051630" in out["prompt"]["user"], "the policy must still see the same evidence"


def test_adapt_row_for_evaluation_matches_the_licensed_path_without_a_repo_id(monkeypatch):
    """eval_targets.cafa_no_knowledge (a source="local" target with no HF repo_id/manifest entry)
    needs this exact prompt shape without going through require_approved_dataset."""
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    licensed = DC.adapt_synthetic_fixture_row(dict(ROW), use="holdout-evaluation")
    unlicensed = DC.adapt_row_for_evaluation(dict(ROW))
    assert unlicensed["prompt"] == licensed["prompt"]
    assert unlicensed["prompt"]["assistant_reasoning"] == ""  # eval never supervises the trace


def test_a_missing_context_source_renders_as_not_available(monkeypatch):
    """The sealed test set has no `ppi_formatted`, but training rows do.

    Dropping the section at eval time would train one prompt shape and score another, and a model
    trained to always see partners is the one that invents them when they vanish. A stable
    placeholder makes "absent" a state the model has seen.
    """
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    no_ppi = {k: v for k, v in ROW.items() if k != "ppi_formatted"}
    user = DC.adapt_synthetic_fixture_row(dict(no_ppi), use="holdout-evaluation")["prompt"]["user"]
    assert "Interaction partners (STRING):\nnot available" in user
    assert user.count("Domain annotations") == 1
    # every section present in both shapes, so the prompt layout is stable across stages
    full = DC.adapt_synthetic_fixture_row(dict(ROW), use="holdout-evaluation")["prompt"]["user"]
    for _, label in DC.CONTEXT_COLUMNS:
        assert label in user and label in full


def test_unusable_sft_rows_are_skipped_not_fatal(monkeypatch):
    """Job 970 died 12 minutes in on one bad row out of ~124k. Sparsity is not a config error."""
    import data

    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    bad = {k: v for k, v in ROW.items() if k != "reasoning"}
    out = data._adapt_or_mark_unusable(
        dict(bad), repo_id="wanglab/bioreason-pro-sft-reasoning-data", use="sft-training")
    assert out["_unusable"] is True
    good = data._adapt_or_mark_unusable(
        dict(ROW), repo_id="wanglab/bioreason-pro-sft-reasoning-data", use="sft-training")
    assert good["_unusable"] is False
    # identical key sets, or datasets.map infers one schema and nulls the real rows
    assert set(out) == set(good)


def test_other_adapter_failures_still_propagate(monkeypatch):
    """Only MissingReasoningEvidence is swallowed. A contract violation must still stop the run."""
    import data

    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    with pytest.raises(ValueError, match="non-empty protein_id and sequence"):
        data._adapt_or_mark_unusable(
            {**ROW, "sequence": ""},
            repo_id="wanglab/bioreason-pro-sft-reasoning-data", use="sft-training")


# --- the path the SFT run actually uses ----------------------------------------------------------

def test_the_real_sft_stream_skips_unsupervisable_rows(monkeypatch):
    """Job 971 died on the path I had not guarded.

    I added skip-and-count to `data.load_sft_dataset`, but `run_sft` streams through
    `train._stream_real_examples`. The guard was real, tested, and on the wrong call site — the
    ADR-008 lesson (grep for the destination, not the declaration) applied to a fix rather than a
    flag. This test drives the actual generator.
    """
    import data
    import train

    rows = [
        {**ROW, "protein_id": "P00001"},
        {k: v for k, v in ROW.items() if k != "reasoning"} | {"protein_id": "P00002"},
        {**ROW, "protein_id": "P00003"},
    ]
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    monkeypatch.setattr(data, "_load_stream", lambda repo: iter(rows))
    monkeypatch.setattr(data, "in_split", lambda pid, split, seed: True)
    monkeypatch.setattr(train, "_adapt_real_row",
                        lambda r, repo: DC.adapt_synthetic_fixture_row(dict(r), use="sft-training"))

    out = list(train._stream_real_examples(train.RunArgs(), train.SFT_DATA_REPO, "rl_train"))
    assert len(out) == 2, "the row without a trace must be skipped, not fatal"
    assert all(o["prompt"]["assistant_reasoning"] for o in out)


def test_a_wholly_unsupervisable_stream_is_refused(monkeypatch):
    """Skipping is for sparsity. Skipping everything means the variant or dataset is wrong."""
    import data
    import train

    bad = {k: v for k, v in ROW.items() if k != "reasoning"}
    rows = [{**bad, "protein_id": f"P{i:05d}"} for i in range(400)]
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    DC._go_ancestors_for_targets.cache_clear()
    monkeypatch.setattr(data, "_load_stream", lambda repo: iter(rows))
    monkeypatch.setattr(data, "in_split", lambda pid, split, seed: True)
    monkeypatch.setattr(train, "_adapt_real_row",
                        lambda r, repo: DC.adapt_synthetic_fixture_row(dict(r), use="sft-training"))

    with pytest.raises(ValueError, match="refusing to train a decimated arm"):
        list(train._stream_real_examples(train.RunArgs(), train.SFT_DATA_REPO, "rl_train"))
