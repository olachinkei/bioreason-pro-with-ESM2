"""Pre-GPU units for the composite reward (bioreason_pro/rewards.py).

Conventions mirror the authoritative public eval (bioreason2 evals/cafa_evals.py).
"""

from bioreason_pro import rewards as R


def test_extract_go_terms_from_whole_text():
    text = "reason GO:1111111 then answer GO:0003674 and GO:0008150"
    assert R.extract_go_terms(text) == {"GO:1111111", "GO:0003674", "GO:0008150"}


def test_extract_go_terms_final_answer_only_splits_on_think():
    text = "<think>GO:1111111 draft</think> final GO:0003674"
    assert R.extract_go_terms(text, final_answer_only=True) == {"GO:0003674"}
    assert "GO:1111111" in R.extract_go_terms(text, final_answer_only=False)


def test_r_format_rewards_think_and_go_term():
    assert R.r_format("<think>r</think> GO:0003674") == 1.0
    assert R.r_format("<think>r</think> no terms") == 0.5   # think only
    assert R.r_format("GO:0003674 only") == 0.5             # term only
    assert R.r_format("nothing here") == 0.0


def test_r_conciseness_is_bounded_and_nonpositive():
    w = R.RewardWeights(len_target_tokens=10, len_penalty_cap=1.0)
    assert R.r_conciseness("a " * 5, w) == 0.0
    assert -1.0 <= R.r_conciseness("a " * 1000, w) <= 0.0


def test_ia_weighted_f1_perfect_and_weighting():
    assert R.ia_weighted_f1({"GO:1"}, {"GO:1"}, None) == 1.0
    assert R.ia_weighted_f1({"GO:1"}, {"GO:2"}, None) == 0.0
    ia = {"GO:hi": 10.0, "GO:lo": 0.1, "GO:miss": 5.0}
    hi = R.ia_weighted_f1({"GO:hi"}, {"GO:hi", "GO:miss"}, ia)
    lo = R.ia_weighted_f1({"GO:lo"}, {"GO:lo", "GO:miss"}, ia)
    assert hi > lo


def test_ia_weighted_f1_zero_weight_terms_no_crash():
    # Predicted terms present but all IA-weight 0 (e.g. GO roots) → no ZeroDivisionError, score 0.
    ia = {"GO:root": 0.0, "GO:real": 5.0}
    assert R.ia_weighted_f1({"GO:root"}, {"GO:real"}, ia) == 0.0
    assert R.ia_weighted_f1({"GO:root"}, {"GO:root"}, ia) == 0.0   # tp weight 0 too


def test_propagate_ancestors_true_path_rule():
    anc = {"GO:child": {"GO:parent", "GO:root"}}
    assert R.propagate_ancestors({"GO:child"}, anc) == {"GO:child", "GO:parent", "GO:root"}


def test_ground_truth_go_handles_lists_and_stringified_lists():
    cols = {"go_mf": [["GO:0003674"], "['GO:0005515', 'GO:0008150']"], "go_bp": [None, ["GO:0009987"]]}
    gts = R._ground_truth_go(cols)
    assert gts[0] == {"GO:0003674"}
    assert gts[1] == {"GO:0005515", "GO:0008150", "GO:0009987"}


def test_make_reward_fn_composite_shape_and_dominance():
    fn = R.make_reward_fn({}, None, R.RewardWeights())
    completions = [
        "<think>t</think> answer GO:0003674",   # matches truth + well-formed
        "GO:9999999",                            # wrong term, no think
    ]
    cols = {"go_mf": ["['GO:0003674']", "['GO:0003674']"]}
    out = fn(["p", "p"], completions, **cols)
    assert len(out) == 2 and out[0] > out[1]


def test_reward_group_smoke_is_finite_and_non_degenerate():
    gt = {"GO:0003674"}
    completions = ["<think>x</think> GO:0003674", "GO:9999999"]
    components = [R.reward_components(text, gt, {}, None) for text in completions]
    values = [item.total for item in components]
    R.validate_reward_group(values)
    assert components[0].fmax == 1.0
    assert components[0].total > components[1].total
