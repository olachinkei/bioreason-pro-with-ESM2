"""CPU-only checks for finite custom-loop training bounds."""

from __future__ import annotations

import pytest

from train import RunArgs


def test_default_streaming_training_is_finite():
    args = RunArgs()

    assert args.streaming_steps_per_epoch == 500
    assert args.resolved_max_steps() == 5_000


def test_explicit_max_steps_overrides_streaming_epoch_bound():
    args = RunArgs(epochs=10, streaming_steps_per_epoch=500, max_steps=7)

    assert args.resolved_max_steps() == 7


def test_subset_and_senpai_epoch_ceiling_reduce_default_bound(monkeypatch):
    monkeypatch.setenv("SENPAI_MAX_EPOCHS", "2")
    args = RunArgs(epochs=10, streaming_steps_per_epoch=500, data_subset_frac=0.25)

    assert args.resolved_max_steps() == 250


@pytest.mark.parametrize("max_steps", [0, -2])
def test_invalid_step_overrides_are_rejected(max_steps):
    with pytest.raises(ValueError, match="max_steps"):
        RunArgs(smoke=True, max_steps=max_steps).validate()


def test_non_positive_senpai_epoch_ceiling_is_rejected(monkeypatch):
    monkeypatch.setenv("SENPAI_MAX_EPOCHS", "0")

    with pytest.raises(ValueError, match="SENPAI_MAX_EPOCHS"):
        RunArgs().resolved_max_steps()
