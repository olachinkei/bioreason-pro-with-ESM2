"""Unit tests for bioreason_pro.baselines — plan.md Phase 2's zero-parameter reference baselines
(pure, no network)."""

from collections import Counter

import pytest

from bioreason_pro.baselines import (
    as_generated_response,
    extract_interpro_ids,
    interpro2go_predict,
    label_prior_terms,
    parse_interpro2go_mapping,
)

SAMPLE_INTERPRO2GO = """\
!version date: 2026/07/06 18:13:03
!description: Mapping of GO terms to InterPro entries.
!
InterPro:IPR000003 Retinoid X receptor/HNF4 > GO:DNA binding ; GO:0003677
InterPro:IPR000003 Retinoid X receptor/HNF4 > GO:nuclear steroid receptor activity ; GO:0003707
InterPro:IPR000006 Metallothionein, vertebrate > GO:metal ion binding ; GO:0046872
""".splitlines()


def test_parse_interpro2go_mapping_groups_by_ipr_id():
    mapping = parse_interpro2go_mapping(SAMPLE_INTERPRO2GO)
    assert mapping == {
        "IPR000003": {"GO:0003677", "GO:0003707"},
        "IPR000006": {"GO:0046872"},
    }


def test_parse_interpro2go_mapping_rejects_unrecognized_line():
    with pytest.raises(ValueError, match="unrecognized interpro2go line"):
        parse_interpro2go_mapping(["not a valid line at all"])


def test_extract_interpro_ids_from_formatted_block():
    formatted = "- IPR000003: some domain [1-50]\n- IPR000006: another domain [60-90]"
    assert extract_interpro_ids(formatted) == {"IPR000003", "IPR000006"}


@pytest.mark.parametrize("value", [None, "", "not available"])
def test_extract_interpro_ids_empty_when_absent(value):
    assert extract_interpro_ids(value) == set()


def test_interpro2go_predict_unions_mapped_terms():
    mapping = parse_interpro2go_mapping(SAMPLE_INTERPRO2GO)
    predicted = interpro2go_predict({"IPR000003", "IPR000006"}, mapping)
    assert predicted == {"GO:0003677", "GO:0003707", "GO:0046872"}


def test_interpro2go_predict_unknown_ipr_id_contributes_nothing():
    mapping = parse_interpro2go_mapping(SAMPLE_INTERPRO2GO)
    assert interpro2go_predict({"IPR999999"}, mapping) == set()


def test_label_prior_terms_picks_top_n_per_aspect_ignoring_protein():
    counts = {
        "go_mf": Counter({"GO:0000001": 10, "GO:0000002": 5, "GO:0000003": 1}),
        "go_bp": Counter({"GO:0000010": 7}),
    }
    prior = label_prior_terms(counts, top_n={"go_mf": 2, "go_bp": 5})
    assert prior == {"go_mf": {"GO:0000001", "GO:0000002"}, "go_bp": {"GO:0000010"}}


def test_label_prior_terms_zero_n_is_empty():
    counts = {"go_mf": Counter({"GO:0000001": 10})}
    assert label_prior_terms(counts, top_n={"go_mf": 0}) == {"go_mf": set()}


def test_as_generated_response_is_deterministic_and_parseable():
    from bioreason_pro.rewards import extract_go_terms

    text = as_generated_response({"GO:0000002", "GO:0000001"})
    assert text == "GO:0000001 GO:0000002"
    assert extract_go_terms(text) == {"GO:0000001", "GO:0000002"}
