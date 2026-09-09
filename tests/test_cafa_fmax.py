"""Pre-GPU units for the cafaeval wrapper (bioreason_pro/cafa_fmax.py).

cafa_eval emits per-namespace best-score frames only; the top-line weighted_fmax is the
caller-computed mean over MF/BP/CC. These guard the aggregation + the dict-of-frames parsing
without needing cafaeval installed at import time.
"""

import math

import pytest

import pandas as pd

from bioreason_pro.cafa_fmax import _per_aspect_fw, aggregate_weighted_fmax


def test_overall_is_mean_of_aspects():
    out = aggregate_weighted_fmax({"mf": 0.8, "bp": 0.7, "cc": 0.6})
    assert out["weighted_fmax_mf"] == 0.8
    assert math.isclose(out["weighted_fmax"], 0.7)


def test_overall_averages_only_present_aspects():
    out = aggregate_weighted_fmax({"mf": 0.9, "bp": 0.7})
    assert math.isclose(out["weighted_fmax"], 0.8)
    assert "weighted_fmax_cc" not in out


def test_empty_is_nan():
    assert math.isnan(aggregate_weighted_fmax({})["weighted_fmax"])


def test_per_aspect_fw_parses_cafaeval_best_scores_dict():
    # cafa_eval returns best_scores_dict = {metric: DataFrame with columns f, f_w and ns index}
    df = pd.DataFrame(
        {"f": [0.5, 0.6, 0.7], "f_w": [0.81, 0.72, 0.63]},
        index=pd.Index(["molecular_function", "biological_process", "cellular_component"], name="ns"),
    )
    per = _per_aspect_fw({"f_w": df})
    assert per == {"mf": 0.81, "bp": 0.72, "cc": 0.63}


def test_per_aspect_fw_raises_without_fw_column():
    df = pd.DataFrame({"f": [0.5]}, index=pd.Index(["molecular_function"], name="ns"))
    with pytest.raises(KeyError):
        _per_aspect_fw({"f": df})


def test_aggregate_reports_how_many_aspects_it_averaged():
    """A model predicting one aspect is scored on one aspect — the denominator must be visible."""
    from bioreason_pro.cafa_fmax import aggregate_weighted_fmax

    three = aggregate_weighted_fmax({"mf": 0.3, "bp": 0.2, "cc": 0.1})
    assert three["weighted_fmax_n_aspects"] == 3.0
    assert three["weighted_fmax"] == pytest.approx(0.2)

    # cafaeval returns nothing for a namespace with no predictions at all.
    one = aggregate_weighted_fmax({"mf": 0.48})
    assert one["weighted_fmax_n_aspects"] == 1.0
    # The metric of record is unchanged: still the mean over what was returned.
    assert one["weighted_fmax"] == pytest.approx(0.48)
    # Which is exactly why it must not be ranked against the three-aspect number.
    assert one["weighted_fmax"] > three["weighted_fmax"]


def test_sweep_runner_flags_an_incomparable_arm():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "compare_prompt_variants.py").read_text(
        encoding="utf-8"
    )
    assert "NOT COMPARABLE" in src
    assert "weighted_fmax_n_aspects" in src
