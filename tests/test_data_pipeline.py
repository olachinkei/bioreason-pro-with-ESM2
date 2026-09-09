"""Pre-GPU units for the CAFA5 data pipeline pure functions (data.py). No HF/GPU needed."""

import data as D
from bioreason_pro.special_tokens import GO_GRAPH_PAD_TOKEN, PROTEIN_PAD_TOKEN


def _example():
    return {
        "protein_id": "P12345",
        "sequence": "MKTAYIAK",   # len 8
        "prompt": {
            "system": " You are a protein function expert. ",
            "user": "Predict GO terms for organism Human.",
            "assistant_reasoning": "  The sequence has a kinase motif.  ",
            "assistant_answer": "GO:0004672",
        },
        "go_aspect": "MF",
        "ground_truth_go_terms": "GO:0004672",
    }


def test_format_folds_system_and_keeps_reasoning_sibling():
    out = D.format_cafa5_for_protein_llm(_example())
    user, assistant = out["prompt"]
    # user content is ordered [protein, go_graph, text]; system folded into the text item
    assert [c["type"] for c in user["content"]] == ["protein", "go_graph", "text"]
    assert user["content"][2]["text"] == "You are a protein function expert.\n\nPredict GO terms for organism Human."
    # reasoning_content is a SIBLING of content, not inside it
    assert assistant["reasoning_content"] == "The sequence has a kinase motif."
    assert assistant["content"] == [{"type": "text", "text": "GO:0004672"}]
    assert out["protein_sequences"] == ["MKTAYIAK"] and out["answer"] == "GO:0004672"


def test_expand_pad_tokens_counts():
    text = f"a {PROTEIN_PAD_TOKEN} b {GO_GRAPH_PAD_TOKEN} c"
    out = D.expand_pad_tokens(text, "MKTAYIAK", max_length_protein=1024, num_go_tokens=200)
    assert out.count(PROTEIN_PAD_TOKEN) == 8 + 2          # min(len,cap) + 2 (ESM BOS/EOS)
    assert out.count(GO_GRAPH_PAD_TOKEN) == 200           # fixed GO block
    # protein cap applies
    out2 = D.expand_pad_tokens(text, "M" * 5000, max_length_protein=1024, num_go_tokens=10)
    assert out2.count(PROTEIN_PAD_TOKEN) == 1024 + 2 and out2.count(GO_GRAPH_PAD_TOKEN) == 10


def test_mask_assistant_labels_only_between_markers_and_pad():
    astart, aend, pad = [10, 11], [12], 0
    ids = [1, 2, 10, 11, 5, 6, 7, 12, 99, 10, 11, 8, 12, pad, pad]
    labels = D.mask_assistant_labels(ids, astart, aend, pad)
    #                idx: 0  1   2   3  4  5  6   7   8   9  10  11  12  13 14
    expected = [-100, -100, -100, -100, 5, 6, 7, -100, -100, -100, -100, 8, -100, -100, -100]
    assert labels == expected


def test_mask_ignores_trailing_assistant_header_without_end():
    astart, aend, pad = [10, 11], [12], 0
    ids = [10, 11, 5, 12, 10, 11]        # trailing header has no following end marker
    assert D.mask_assistant_labels(ids, astart, aend, pad) == [-100, -100, 5, -100, -100, -100]


def test_in_split_matches_protein_split():
    for pid in ("A0A1", "P99999", "Q8N1"):
        s = D.protein_split(pid, 0)
        assert D.in_split(pid, s, 0) and sum(D.in_split(pid, x, 0) for x in D.SPLITS) == 1
