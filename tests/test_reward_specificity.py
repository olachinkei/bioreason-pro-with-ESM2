"""The last un-run ledger condition: penalise GO ids that another named id already implies.

Scoring propagates predictions, so naming an ancestor earns nothing propagation would not add for
free — deleting them from the Phase 4 rollouts moved weighted F_max by exactly 0.000000. They are not
free in *generation budget*: at the 64-token operating point the model emits ~5 ids, so each redundant
one displaces a leaf. `aspect_mean_specific` makes that displacement visible to the optimiser.

The penalty is bounded in [0,1] and weighted at 0.1 (matching lambda_fmt), so it cannot outweigh the
fmax proxy it is attached to. lambda_spec defaults to 0.0, so both shipped variants are bit-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bioreason_pro import rewards as R

ROOT = Path(__file__).resolve().parents[1]

# GO:0000101 implies GO:0000100; GO:0000201 implies GO:0000200.
ANCESTORS = {
    "GO:0000101": {"GO:0000100"}, "GO:0000100": set(),
    "GO:0000201": {"GO:0000200"}, "GO:0000200": set(),
}


def test_the_variant_is_registered_and_fails_closed(monkeypatch):
    assert "aspect_mean_specific" in R.REWARD_VARIANTS
    monkeypatch.setenv("SENPAI_REWARD_VARIANT", "aspect_mean_specific")
    assert R.active_reward_variant() == "aspect_mean_specific"


def test_all_specific_predictions_carry_no_redundancy():
    assert R.ancestor_redundancy("GO:0000101 GO:0000201", ANCESTORS) == 0.0


def test_naming_an_implied_ancestor_is_counted():
    """Two ids of which one is implied by the other -> half the emission is redundant."""
    assert R.ancestor_redundancy("GO:0000101 GO:0000100", ANCESTORS) == pytest.approx(0.5)


def test_redundancy_is_bounded_and_empty_is_not_punished():
    assert R.ancestor_redundancy("no terms here", ANCESTORS) == 0.0
    # Every non-leaf named alongside its leaf: 2 of 3 redundant, still <= 1.
    r = R.ancestor_redundancy("GO:0000101 GO:0000100 GO:0000201 GO:0000200", ANCESTORS)
    assert 0.0 <= r <= 1.0


def test_default_weights_leave_the_shipped_total_bit_identical():
    """lambda_spec=0.0 must not change the total — but redundancy is still MEASURED."""
    completion = "<think>x</think> GO:0000101 GO:0000100"
    plain = R.reward_components(completion, {"GO:0000101"}, ANCESTORS, None)
    expected = plain.fmax + 0.1 * plain.format + 0.05 * plain.conciseness
    assert plain.total == pytest.approx(expected)


def test_redundancy_is_measured_even_when_it_is_not_penalised():
    """A constant 0.0 on unpenalised runs is a placeholder, not a baseline to compare against."""
    completion = "<think>x</think> GO:0000101 GO:0000100"
    plain = R.reward_components(completion, {"GO:0000101"}, ANCESTORS, None)
    assert plain.redundancy == pytest.approx(0.5), (
        "unpenalised runs must still report the redundancy they actually emitted"
    )


def test_the_penalty_lowers_the_total_only_when_enabled():
    completion = "<think>x</think> GO:0000101 GO:0000100"
    true = {"GO:0000101"}
    off = R.reward_components(completion, true, ANCESTORS, None)
    on = R.reward_components(
        completion, true, ANCESTORS, None, R.RewardWeights(lambda_spec=0.1)
    )
    # fmax is unchanged — propagation makes the two prediction sets identical.
    assert on.fmax == pytest.approx(off.fmax)
    assert on.redundancy == pytest.approx(0.5)
    assert on.total == pytest.approx(off.total - 0.1 * 0.5)


def test_a_specific_answer_now_outscores_a_redundant_one_of_equal_fmax():
    """The whole point: equal score under the metric, different reward."""
    true = {"GO:0000101"}
    w = R.RewardWeights(lambda_spec=0.1)
    specific = R.reward_components("<think>x</think> GO:0000101", true, ANCESTORS, None, w)
    padded = R.reward_components(
        "<think>x</think> GO:0000101 GO:0000100", true, ANCESTORS, None, w
    )
    assert specific.fmax == pytest.approx(padded.fmax)
    assert specific.total > padded.total


def test_train_wires_the_variant_and_logs_its_component():
    src = (ROOT / "train.py").read_text(encoding="utf-8")
    # The literal tuple this used to pin moved into rewards.ASPECT_AWARE_VARIANTS when a third
    # aspect-aware variant arrived (ADR-014); assert the guarantee, not the spelling. Membership is
    # checked in tests/test_reasoning_reward.py.
    assert "rewards.ASPECT_AWARE_VARIANTS" in src, "go_aspects must load for every aspect-aware variant"
    assert "aspect_mean_specific" in R.ASPECT_AWARE_VARIANTS
    assert "aspect_mean" in R.ASPECT_AWARE_VARIANTS
    assert "RewardWeights(lambda_spec=0.1)" in src
    assert '"rl/reward_redundancy"' in src, "a new reward term must be observable in W&B"
    assert "lambda_spec=" in src, "the active weight must be printed for the run record"
