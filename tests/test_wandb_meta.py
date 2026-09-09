"""Contract tests for bioreason_pro.wandb_meta — the shared derived-tags/note logic.

Every `wandb.init()` call site now passes `tags=derived_tags(...)` and `notes=build_note(...)`, and
`scripts/apply_run_tags.py` imports the same `derived_tags`/`provenance` rather than keeping its own
copy. These tests pin the one implementation both paths share.
"""

from __future__ import annotations

from pathlib import Path

from bioreason_pro import wandb_meta as M

ROOT = Path(__file__).resolve().parent.parent


def test_provenance_flags_any_incompatible_marker_anywhere_in_config():
    for marker in ("esm3", "ESM3", "esm-c", "ESMC", "esmc_600m"):
        assert M.provenance({"protein_model_name": marker}) == "pre-boundary", marker


def test_provenance_defaults_to_approved():
    assert M.provenance({"protein_model_name": "facebook/esm2_t33_650M_UR50D"}) == "approved"
    assert M.provenance({}) == "approved"


def test_derived_tags_adds_stage_for_known_job_types():
    assert M.derived_tags("sft-training", {}) == ["approved", "sft"]
    assert M.derived_tags("rl-training", {}) == ["approved", "rl"]
    assert M.derived_tags("full-holdout-evaluation", {}) == ["approved", "eval"]
    assert M.derived_tags("agent-iteration-finalize", {}) == ["agent", "approved"]


def test_derived_tags_has_no_stage_tag_for_an_unknown_job_type():
    tags = M.derived_tags("some-future-job-type", {})
    assert tags == ["approved"]


def test_derived_tags_flags_pre_boundary_regardless_of_job_type():
    tags = M.derived_tags("sft-training", {"esm_model_name": "esm3_sm_open_v1"})
    assert tags == ["pre-boundary", "sft"]


def test_build_note_opens_with_the_human_label():
    assert M.build_note("sft-training", {}).startswith("SFT")
    assert M.build_note("rl-training", {}).startswith("RL")
    assert M.build_note("full-holdout-evaluation", {}).startswith("sealed holdout")


def test_build_note_falls_back_to_the_raw_job_type_when_unlabelled():
    assert M.build_note("some-future-job-type", {}) == "some-future-job-type"
    assert M.build_note(None, {}) == "run"


def test_build_note_surfaces_known_fields_in_order_and_skips_empty_ones():
    note = M.build_note(
        "rl-training",
        {"target_variant": "leaf_only_reasoned", "reward_variant": "", "train_seed": 2},
    )
    assert note == "RL · target=leaf_only_reasoned · seed=2"


def test_build_note_is_a_single_line_orientation_not_a_config_dump():
    config = {
        "target_variant": "leaf_only_reasoned",
        "reward_variant": "aspect_mean_reasoned",
        "train_seed": 1,
        "learning_rate": 1e-5,  # not in the field list — must not appear
        "lora_r": 128,          # not in the field list — must not appear
    }
    note = M.build_note("rl-training", config)
    assert "\n" not in note
    assert "learning_rate" not in note
    assert "lora_r" not in note
    assert "target=leaf_only_reasoned" in note
    assert "reward=aspect_mean_reasoned" in note
    assert "seed=1" in note


def test_build_note_appends_the_extra_clause_last():
    note = M.build_note("model-evaluation", {"target": "bioreason_pro_test"}, extra="PR #170")
    assert note.endswith("PR #170")
    assert note.startswith("eval")


def test_build_note_truncates_absurdly_long_notes():
    config = {"artifact": "x" * 1000}
    note = M.build_note("model-evaluation", config)
    assert len(note) <= M._NOTE_MAX_LEN
    assert note.endswith("…")


def test_apply_run_tags_uses_the_shared_module_not_a_private_copy():
    """Job 691 taught this project to grep for a flag's destination, not its declaration (ADR-008).
    Same principle here: assert the SHARED implementation is what the script imports."""
    text = (ROOT / "scripts" / "apply_run_tags.py").read_text(encoding="utf-8")
    assert "from bioreason_pro.wandb_meta import" in text
    assert "def _provenance" not in text, "a private duplicate would drift from the shared one"
