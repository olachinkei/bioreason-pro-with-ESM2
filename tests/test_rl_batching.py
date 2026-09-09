"""ADR-036/037: batching multiple proteins per RL step, with pooled-std advantage normalization.

Two things must hold at once:

1. The defaults (`proteins_per_step=1`, `rl_advantage_normalization="group_mean"`) reproduce the
   pre-ADR-037 Dr.GRPO baseline bit-for-bit — this is a strict superset, not a behaviour change for
   anyone who doesn't touch the new flags.
2. The paper's own formula (Eq. 4.19-4.20) — per-protein group mean, pooled batch std — is correct,
   including the degenerate all-rewards-identical case, which must not raise or produce a NaN.
"""

from __future__ import annotations

import torch

import train


def test_group_mean_mode_reproduces_the_pre_existing_dr_grpo_baseline():
    """`group_mean` (the default) must be exactly `r - r.mean()`, no std division at all."""
    groups = [torch.tensor([0.1, 0.4, 0.2, 0.9]), torch.tensor([0.5, 0.5, 0.5, 0.5])]
    advantages, global_std = train._pooled_advantages(groups, "group_mean")
    for r, adv in zip(groups, advantages):
        assert torch.allclose(adv, r - r.mean())
    # global_std is still measured (for logging/diagnostics) even in group_mean mode — it's simply
    # not used as the advantage denominator there.
    expected_std = float(torch.cat(groups).std(correction=0))
    assert float(global_std) == expected_std


def test_global_std_mode_centres_per_protein_but_divides_by_the_pooled_std():
    g0 = torch.tensor([0.1, 0.3, 0.5, 0.7])
    g1 = torch.tensor([0.2, 0.2, 0.4, 0.4])
    advantages, global_std = train._pooled_advantages([g0, g1], "group_mean_global_std")

    all_r = torch.cat([g0, g1])
    expected_std = float(all_r.std(correction=0))
    assert float(global_std) == expected_std

    denom = expected_std + train.ADV_STD_EPS
    assert torch.allclose(advantages[0], (g0 - g0.mean()) / denom)
    assert torch.allclose(advantages[1], (g1 - g1.mean()) / denom)


def test_global_std_mode_at_a_single_protein_reproduces_the_papers_formula_directly():
    """B=1 (one protein's own G rollouts) with global-std mode is just vanilla GRPO on that group —
    the same per-protein-group std the paper's own formula reduces to at that group size."""
    r = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
    advantages, global_std = train._pooled_advantages([r], "group_mean_global_std")
    denom = float(r.std(correction=0)) + train.ADV_STD_EPS
    assert torch.allclose(advantages[0], (r - r.mean()) / denom)


def test_a_wholly_degenerate_batch_does_not_nan():
    """global_std==0 forces every per-protein numerator to 0 too (a constant batch implies every
    subgroup mean equals that same constant), so this must be a harmless 0/eps, never a NaN."""
    groups = [torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])]
    advantages, global_std = train._pooled_advantages(groups, "group_mean_global_std")
    assert float(global_std) == 0.0
    for adv in advantages:
        assert torch.allclose(adv, torch.zeros_like(adv))
        assert torch.isfinite(adv).all()


def test_unknown_normalization_mode_fails_closed():
    import pytest

    with pytest.raises(ValueError, match="unknown rl_advantage_normalization"):
        train._pooled_advantages([torch.tensor([0.1, 0.2])], "not_a_real_mode")


def test_run_args_validates_proteins_per_step_and_normalization_mode():
    import pytest

    args = train.RunArgs(smoke=True, proteins_per_step=0)
    with pytest.raises(ValueError, match="proteins_per_step must be >= 1"):
        args.validate(require_training_dataset=False)

    args = train.RunArgs(smoke=True, rl_advantage_normalization="bogus")
    with pytest.raises(ValueError, match="rl_advantage_normalization"):
        args.validate(require_training_dataset=False)


def test_run_args_defaults_are_the_pre_adr037_behaviour():
    args = train.RunArgs()
    assert args.proteins_per_step == 1
    assert args.rl_advantage_normalization == "group_mean"
