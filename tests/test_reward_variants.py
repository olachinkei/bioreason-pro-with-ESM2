"""The RL reward must be able to aggregate the way the metric does.

The shipped reward is one IA-weighted F1 over the union of go_mf/go_bp/go_cc, while the evaluation
metric is the MEAN of per-aspect F_max. That mismatch let GRPO raise the union F1 by piling into the
cheapest aspect: CC has 4,043 terms against MF's 11,263 and BP's 27,942, and at 100 steps CC nearly
doubled while MF fell 27%, leaving the aspect-mean metric flat. `aspect_mean` closes the gap.
"""

from __future__ import annotations

import pytest

from bioreason_pro import rewards as R

# One two-level chain per aspect, so propagation has something to do in each.
ANCESTORS = {
    "GO:0000101": {"GO:0000100"}, "GO:0000100": set(),   # MF
    "GO:0000201": {"GO:0000200"}, "GO:0000200": set(),   # BP
    "GO:0000301": {"GO:0000300"}, "GO:0000300": set(),   # CC
}
ASPECTS = {
    "GO:0000100": "MF", "GO:0000101": "MF",
    "GO:0000200": "BP", "GO:0000201": "BP",
    "GO:0000300": "CC", "GO:0000301": "CC",
}
ALL_TRUE = {"GO:0000101", "GO:0000201", "GO:0000301"}


def test_default_variant_is_the_shipped_union(monkeypatch):
    monkeypatch.delenv("SENPAI_REWARD_VARIANT", raising=False)
    assert R.active_reward_variant() == "union"


def test_unknown_variant_fails_closed(monkeypatch):
    monkeypatch.setenv("SENPAI_REWARD_VARIANT", "make_it_better")
    with pytest.raises(ValueError, match="not a known reward variant"):
        R.active_reward_variant()


def test_omitting_go_aspects_keeps_the_union_reward_bit_for_bit():
    completion = "<think>x</think> GO:0000101 GO:0000201"
    union = R.reward_components(completion, ALL_TRUE, ANCESTORS, None)
    direct = R.r_fmax(completion, ALL_TRUE, ANCESTORS, None)
    assert union.fmax == direct


def test_aspect_mean_averages_only_the_aspects_present_in_the_truth():
    """A protein annotated in one aspect must not be scored against three."""
    completion = "GO:0000101"
    only_mf = {"GO:0000101"}
    assert R.r_fmax_aspect_mean(completion, only_mf, ANCESTORS, None, ASPECTS) == 1.0


def test_aspect_mean_punishes_dropping_an_aspect_that_union_would_forgive():
    """The whole point: getting two of three aspects perfectly is 2/3, not ~2/3-weighted-by-mass."""
    # Predict MF and BP exactly, miss CC entirely.
    completion = "GO:0000101 GO:0000201"
    aspect_mean = R.r_fmax_aspect_mean(completion, ALL_TRUE, ANCESTORS, None, ASPECTS)
    assert aspect_mean == pytest.approx(2 / 3)

    # The union reward sees one pooled set and scores higher, because the missing CC terms are only
    # a fraction of the pooled mass rather than a whole third of the score.
    union = R.r_fmax(completion, ALL_TRUE, ANCESTORS, None)
    assert union > aspect_mean


def test_aspect_mean_cannot_be_gamed_by_piling_into_one_cheap_aspect():
    """Flooding CC while losing MF must not look free, which is what the union reward allowed."""
    balanced = "GO:0000101 GO:0000201 GO:0000301"
    cc_heavy = "GO:0000300 GO:0000301 GO:0000201"  # CC over-covered, MF dropped
    assert R.r_fmax_aspect_mean(balanced, ALL_TRUE, ANCESTORS, None, ASPECTS) > (
        R.r_fmax_aspect_mean(cc_heavy, ALL_TRUE, ANCESTORS, None, ASPECTS)
    )


def test_reward_components_routes_to_aspect_mean_when_aspects_are_given():
    completion = "GO:0000101 GO:0000201"
    with_aspects = R.reward_components(
        completion, ALL_TRUE, ANCESTORS, None, go_aspects=ASPECTS
    )
    assert with_aspects.fmax == pytest.approx(2 / 3)
    # The other components are untouched by the variant.
    plain = R.reward_components(completion, ALL_TRUE, ANCESTORS, None)
    assert with_aspects.format == plain.format
    assert with_aspects.conciseness == plain.conciseness


def test_train_wires_the_variant_and_the_tracer_accepts_it():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    train_src = (root / "train.py").read_text(encoding="utf-8")
    assert "rewards.active_reward_variant()" in train_src
    assert "go_aspects=go_aspects" in train_src
    tracing_src = (root / "bioreason_pro" / "weave_tracing.py").read_text(encoding="utf-8")
    assert "go_aspects: dict[str, str] | None = None" in tracing_src
    assert "go_aspects=aspects" in tracing_src
