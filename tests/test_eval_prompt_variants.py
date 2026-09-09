"""The eval prompt must be byte-identical to training unless a variant is asked for explicitly."""

from __future__ import annotations

import pytest

from eval_targets import base


def test_default_is_byte_identical_to_the_training_instruction(monkeypatch):
    monkeypatch.delenv("SENPAI_EVAL_PROMPT_VARIANT", raising=False)
    assert base.active_prompt_variant() == "baseline"
    assert base.user_instruction() == base.USER_INSTRUCTION


def test_specific_only_appends_guidance_without_altering_the_original_text(monkeypatch):
    monkeypatch.setenv("SENPAI_EVAL_PROMPT_VARIANT", "specific_only")
    text = base.user_instruction()
    assert text.startswith(base.USER_INSTRUCTION)
    assert "most specific GO terms" in text
    assert text != base.USER_INSTRUCTION


def test_unknown_variant_fails_closed_rather_than_silently_using_the_baseline(monkeypatch):
    monkeypatch.setenv("SENPAI_EVAL_PROMPT_VARIANT", "make_it_better")
    with pytest.raises(ValueError, match="not a known prompt variant"):
        base.active_prompt_variant()
    with pytest.raises(ValueError, match="unknown prompt variant"):
        base.user_instruction("make_it_better")


def test_explicit_argument_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("SENPAI_EVAL_PROMPT_VARIANT", "specific_only")
    assert base.user_instruction("baseline") == base.USER_INSTRUCTION


def test_adapt_row_uses_the_selected_variant(monkeypatch):
    monkeypatch.setenv("SENPAI_EVAL_PROMPT_VARIANT", "specific_only")
    target = base.EvalTarget(
        name="unit_target",
        hf_repo="unit/repo",
        context_fields=(("Organism", "organism"),),
    )
    row = {"protein_id": "P1", "sequence": "MKT", "organism": "E. coli"}
    adapted = base.adapt_row(row, target)
    assert "most specific GO terms" in adapted["prompt"]["user"]
    assert adapted["prompt"]["user"].startswith(base.USER_INSTRUCTION)


def test_adapt_row_default_carries_no_variant_text(monkeypatch):
    monkeypatch.delenv("SENPAI_EVAL_PROMPT_VARIANT", raising=False)
    target = base.EvalTarget(
        name="unit_target",
        hf_repo="unit/repo",
        context_fields=(("Organism", "organism"),),
    )
    row = {"protein_id": "P1", "sequence": "MKT", "organism": "E. coli"}
    assert "most specific" not in base.adapt_row(row, target)["prompt"]["user"]


def _user_turn():
    return {
        "role": "user",
        "content": [
            {"type": "protein", "text": None},
            {"type": "go_graph", "text": None},
            {"type": "text", "text": "Predict the GO terms."},
        ],
    }


def test_val_path_appends_the_suffix_to_the_text_item_only():
    from train import _append_prompt_suffix

    out = _append_prompt_suffix(_user_turn(), base.PROMPT_VARIANTS["specific_only"])
    assert [item["type"] for item in out["content"]] == ["protein", "go_graph", "text"]
    assert out["content"][0]["text"] is None
    assert out["content"][2]["text"].startswith("Predict the GO terms.")
    assert "most specific GO terms" in out["content"][2]["text"]


def test_val_path_is_untouched_without_a_variant():
    from train import _append_prompt_suffix

    turn = _user_turn()
    assert _append_prompt_suffix(turn, "") is turn


def test_val_path_refuses_a_turn_with_no_text_item():
    from train import _append_prompt_suffix

    turn = {"role": "user", "content": [{"type": "protein", "text": None}]}
    with pytest.raises(ValueError, match="no text item"):
        _append_prompt_suffix(turn, "some guidance")
