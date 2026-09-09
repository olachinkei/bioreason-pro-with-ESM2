"""Contract tests for the ADR-014 reasoning reward.

Two things must hold at once:

1. Every pre-existing variant keeps its EXACT arithmetic, so the recorded numbers for union,
   aspect_mean and aspect_mean_specific stay comparable to their published values.
2. The new terms are actually live — they vary across plausible rollouts. ADR-013 is the cautionary
   tale: `r_format` carried a 0.1 weight for eight phases while scoring an identical 1.0 in all 472
   stored rollouts, contributing exactly nothing to a GRPO advantage that is centred within-group.
   A reward term that cannot vary is decoration, so these tests assert variance, not just presence.
"""

from __future__ import annotations

import pytest

from bioreason_pro import rewards as R

OBO = {"GO:0000002": {"GO:0000001"}, "GO:0000003": set(), "GO:0042803": set()}
IA = None
NAMES = {
    "GO:0042803": "protein homodimerization activity",
    "GO:0000003": "reproduction",
    "GO:0000002": "mitochondrial genome maintenance",
}

EMPTY_THINK = "<think>\n\n</think>\n\nMF: GO:0042803\nBP: GO:0000003"
REASONED = (
    "<think>\nThe sequence carries a coiled-coil segment consistent with homodimerization, and the "
    "C-terminal region resembles known mitochondrial targeting peptides, which points at "
    "mitochondrial genome maintenance rather than a cytosolic role.\n</think>\n\n"
    "MF: GO:0042803\nBP: GO:0000002"
)
LONG_BUT_UNFAITHFUL = (
    "<think>\n" + ("This protein is interesting and clearly important in several ways. " * 12) +
    "\n</think>\n\nMF: GO:0042803\nBP: GO:0000002"
)
# Shaped like the real thing: 466 of 472 stored rollouts carried complete ids and then ran out of
# budget partway through one more. A fragment-only completion would make `r_format` vary and quietly
# invalidate the ADR-013 assertion below.
TRUNCATED = (
    "<think>\nSome reasoning about the sequence and its likely localisation.\n</think>\n\n"
    "MF: GO:0042803\nBP: GO:0000002\nCC: GO:00"
)


# --- 1. the existing variants are untouched ------------------------------------------------------

@pytest.mark.parametrize("completion", [EMPTY_THINK, REASONED, LONG_BUT_UNFAITHFUL, TRUNCATED])
def test_default_weights_reproduce_the_shipped_total_exactly(completion):
    """With the default weights the reasoning terms must cancel out of `total` bit-for-bit."""
    w = R.RewardWeights()
    assert w.lambda_reason == 0.0 and w.lambda_truncation == 0.0
    c = R.reward_components(completion, {"GO:0042803"}, OBO, IA, w, go_names=NAMES)
    expected = (c.fmax + w.lambda_fmt * c.format + w.lambda_len * c.conciseness
                - w.lambda_spec * c.redundancy)
    assert c.total == expected


def test_reasoning_components_are_measured_even_when_unweighted():
    """Same argument as `redundancy`: an unweighted arm still needs a baseline to compare against."""
    c = R.reward_components(REASONED, {"GO:0042803"}, OBO, IA, R.RewardWeights(), go_names=NAMES)
    assert c.substance > 0.0
    assert c.faithfulness > 0.0


def test_the_new_variant_is_registered_and_unknown_names_still_fail_closed(monkeypatch):
    assert "aspect_mean_reasoned" in R.REWARD_VARIANTS
    assert "aspect_mean_reasoned" in R.REASONED_VARIANTS
    assert "aspect_mean_reasoned" in R.ASPECT_AWARE_VARIANTS
    monkeypatch.setenv("SENPAI_REWARD_VARIANT", "aspect_mean_reasoned")
    assert R.active_reward_variant() == "aspect_mean_reasoned"
    monkeypatch.setenv("SENPAI_REWARD_VARIANT", "reasoned")  # near-miss name
    with pytest.raises(ValueError):
        R.active_reward_variant()


# --- 2. the new terms are live ------------------------------------------------------------------

def test_substance_separates_an_empty_trace_from_a_real_one():
    w = R.RewardWeights()
    assert R.r_reasoning_substance(EMPTY_THINK, w) == 0.0
    assert R.r_reasoning_substance(REASONED, w) > 0.0


def test_substance_saturates_so_padding_cannot_farm_it():
    w = R.RewardWeights()
    padded = "<think>\n" + ("x" * 5000) + "\n</think>\n\nMF: GO:0042803"
    assert R.r_reasoning_substance(padded, w) == 1.0


def test_faithfulness_rewards_a_trace_that_accounts_for_its_own_answer():
    """The term that makes a trace worth showing: length alone must not earn it."""
    faithful = R.r_reasoning_faithfulness(REASONED, NAMES)
    padded = R.r_reasoning_faithfulness(LONG_BUT_UNFAITHFUL, NAMES)
    assert faithful > padded
    assert padded == 0.0, "boilerplate that names nothing must score zero"
    assert faithful == pytest.approx(1.0), "both answer terms are named in the trace"


def test_faithfulness_is_zero_without_a_trace_or_without_an_answer():
    assert R.r_reasoning_faithfulness(EMPTY_THINK, NAMES) == 0.0
    assert R.r_reasoning_faithfulness("<think>\nlots of words here\n</think>\n\nMF:", NAMES) == 0.0


def test_faithfulness_accepts_a_bare_go_id_in_the_trace():
    completion = "<think>\nGO:0042803 is supported by the coiled-coil region.\n</think>\n\nMF: GO:0042803"
    assert R.r_reasoning_faithfulness(completion, {}) == pytest.approx(1.0)


def test_truncation_is_detected_and_penalised():
    assert R.r_truncated(TRUNCATED) == 1.0
    assert R.r_truncated(REASONED) == 0.0
    w = R.RewardWeights(lambda_reason=0.25, lambda_truncation=0.1)
    cut = R.reward_components(TRUNCATED, {"GO:0042803"}, OBO, IA, w, go_names=NAMES)
    whole = R.reward_components(REASONED, {"GO:0042803"}, OBO, IA, w, go_names=NAMES)
    assert cut.total < whole.total


def test_substance_and_faithfulness_multiply_rather_than_add():
    """Either one alone is worthless; summing would pay for a long empty trace."""
    w = R.RewardWeights(lambda_reason=0.25)
    padded = R.reward_components(LONG_BUT_UNFAITHFUL, {"GO:0042803"}, OBO, IA, w, go_names=NAMES)
    plain = R.reward_components(LONG_BUT_UNFAITHFUL, {"GO:0042803"}, OBO, IA,
                                R.RewardWeights(), go_names=NAMES)
    assert padded.total == pytest.approx(plain.total), "long but unfaithful earns nothing extra"


def test_the_reasoned_reward_is_not_degenerate_across_a_plausible_group():
    """The ADR-013 check, as a test: this group must not collapse to one value.

    `r_format` would return 1.0 for every member of this group, which is exactly why it taught
    nothing. The reasoning terms must do better on the same input.
    """
    group = [EMPTY_THINK, REASONED, LONG_BUT_UNFAITHFUL, TRUNCATED]
    w = R.RewardWeights(lambda_reason=0.25, lambda_truncation=0.1)
    totals = [R.reward_components(c, {"GO:0042803"}, OBO, IA, w, go_names=NAMES).total
              for c in group]
    R.validate_reward_group(totals, require_non_degenerate=True)
    assert len({R.r_format(c) for c in group}) == 1, "r_format is still constant — the ADR-013 fact"
    assert len(set(totals)) > 1, "the reasoned reward must separate these rollouts"


# --- ADR-027 / plan.md Phase 6: aspect_mean_reasoned trains with the corrected reward -------------

def test_aspect_mean_reasoned_zeroes_the_unpinned_format_and_length_terms():
    """train.py's own per-variant weight selection, pinned by a test rather than only readable
    inline: the paper's RL reward has no format/conciseness term, so Phase 6's variant must not
    either, while every pre-existing variant keeps its historical, non-zero values untouched."""
    import train

    reasoned = train._reward_weights_for_variant("aspect_mean_reasoned")
    assert reasoned.lambda_fmt == 0.0
    assert reasoned.lambda_len == 0.0
    assert reasoned.lambda_reason == 0.25
    assert reasoned.lambda_truncation == 0.1

    for variant in ("union", "aspect_mean", "aspect_mean_specific"):
        w = train._reward_weights_for_variant(variant)
        assert w.lambda_fmt == R.RewardWeights().lambda_fmt
        assert w.lambda_len == R.RewardWeights().lambda_len
