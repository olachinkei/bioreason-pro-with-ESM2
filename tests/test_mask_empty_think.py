"""ADR-015: the SFT loss must stop supervising the manufactured empty <think></think>.

The chat template renders any assistant target without `</think>` as
`<think>\\n\\n</think>\\n\\n` + the answer, so every supervised example was teaching the model to
open and immediately close its reasoning block. It learned that exactly — 472 of 472 stored rollouts
and 16 of 16 in the Phase 11 RL smoke emitted an empty trace, at budgets from 64 to 512. RL could
not undo it: GRPO reweights only what the policy samples.

These tests pin (a) that the default is byte-identical to the shipped supervision, and (b) that the
knob is fail-closed, because a silently-ignored knob turns a mis-specified experiment into something
that looks like a completed one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import train

ROOT = Path(__file__).resolve().parent.parent


class FakeTok:
    """Character-level stand-in: each token is one character, so spans are easy to reason about."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_the_empty_think_span_is_masked_and_the_answer_is_not():
    rendered = "<think>\n\n</think>\n\nMF: GO:0042803"
    ids = _ids(rendered)
    labels = list(ids)  # as if the whole assistant turn were supervised
    out, masked = train._mask_empty_think_labels(ids, labels, FakeTok())

    assert masked == len("<think>\n\n</think>")
    span_end = rendered.index("</think>") + len("</think>")
    assert all(v == -100 for v in out[:span_end]), "the manufactured trace must not be supervised"
    answer_start = rendered.index("MF:")
    assert out[answer_start:] == ids[answer_start:], "the answer must still be supervised"


def test_already_masked_positions_are_not_double_counted():
    rendered = "<think>\n\n</think>\n\nMF: GO:0042803"
    ids = _ids(rendered)
    labels = [-100] * len(ids)
    _out, masked = train._mask_empty_think_labels(ids, labels, FakeTok())
    assert masked == 0, "nothing was supervised, so nothing was un-taught"


def test_a_non_empty_trace_is_left_alone():
    """Only the EMPTY block is the artefact. A real trace must stay supervised if one ever appears."""
    rendered = "<think>\nthe C-terminus resembles a targeting peptide\n</think>\n\nMF: GO:0042803"
    ids = _ids(rendered)
    labels = list(ids)
    out, masked = train._mask_empty_think_labels(ids, labels, FakeTok())
    assert masked == 0
    assert out == labels


def test_default_is_off_so_every_recorded_sft_reproduces():
    assert train.RunArgs().mask_empty_think is False


def test_the_knob_fails_closed_when_the_span_is_absent():
    """A knob that silently does nothing turns a mis-specified run into a plausible-looking result."""
    src = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "mask_empty_think=True but the empty <think></think> span was not found" in src
    assert re.search(r"if n_think_masked == 0:\s*\n\s*raise ValueError", src), \
        "the zero-match case must raise, not warn"


def test_both_sft_launchers_forward_the_knob():
    """Same rule as the train-seed knob: the smoke must be able to set what the full run varies."""
    for name in ("sft.sbatch", "sft_smoke.sbatch"):
        text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'SFT_MASK_EMPTY_THINK="${SFT_MASK_EMPTY_THINK:-false}"' in text, name
        assert '--mask_empty_think "$SFT_MASK_EMPTY_THINK"' in text, name


def test_the_span_constant_matches_what_the_template_actually_renders():
    """If the template changes its spacing, this test fails before a 3.5-hour run does."""
    template = (ROOT / "bioreason_pro" / "qwen3_4b_chat_template.jinja2").read_text(encoding="utf-8")
    assert "'\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'" in template, (
        "the final-assistant branch no longer renders <think>\\n\\n</think> for an empty trace; "
        "train.EMPTY_THINK_SPAN needs to be updated to match"
    )
    assert train.EMPTY_THINK_SPAN == "<think>\n\n</think>"


@pytest.mark.parametrize("flag", [True, False])
def test_run_args_accepts_the_flag(flag):
    assert train.RunArgs(mask_empty_think=flag).mask_empty_think is flag
