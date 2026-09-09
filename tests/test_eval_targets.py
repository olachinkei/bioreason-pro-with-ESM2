"""Unit tests for the eval_targets package — pure (no HF/GPU).

Covers the schema-adaptation seam that lets a raw held-out row (no `prompt` dict) flow into the
protected data.format_cafa5_for_protein_llm, the ground-truth parsing, the has_ppi/has_interpro
gating, split_mode="all" (no in_split filter), and the availability gate on cafa5.
"""

import dataclasses
import types
from pathlib import Path

import pytest

import data
import eval_targets as et
from eval_targets.base import EvalTarget, adapt_row, parse_gt, stream_eval_records


def _test_data_row(**over):
    """A row shaped like a real wanglab/bioreason-pro-test-data row (no `prompt`, no ppi_formatted)."""
    row = {
        "protein_id": "P123",
        "protein_names": "Toxin X",
        "organism": "Homo sapiens",
        "subcellular_location": "Secreted",
        "interpro_formatted": "- IPR000001: some domain [1-50]",
        "go_pred": "Molecular Function (MF): GO:0003674 (molecular function)",
        "sequence": "MDYQRLLFLF",
        "go_mf": "['GO:0003674', 'GO:0098772']",
        "go_bp": None,                      # real rows are None when the aspect is empty
        "go_cc": "['GO:0005575']",
    }
    row.update(over)
    return row


# --------------------------------------------------------------- parse_gt

def test_parse_gt_unions_aspects_and_tolerates_none():
    gt = parse_gt(_test_data_row(), et.get_target("bioreason_pro_test"))
    assert gt == {"GO:0003674", "GO:0098772", "GO:0005575"}


def test_parse_gt_empty_when_all_missing():
    row = {"protein_id": "P0", "sequence": "M"}
    assert parse_gt(row, et.get_target("bioreason_pro_test")) == set()


def test_parse_gt_handles_real_list_and_freetext():
    tgt = EvalTarget(name="t", hf_repo="r", gt_columns=("go_mf",))
    assert parse_gt({"go_mf": ["GO:0003674", "x"]}, tgt) == {"GO:0003674"}
    assert parse_gt({"go_mf": "text GO:0003674 and GO:0005575 here"}, tgt) == {"GO:0003674", "GO:0005575"}


# --------------------------------------------------------------- adapt_row

def test_approved_holdout_prompt_excludes_all_unreviewed_context_columns():
    row = _test_data_row(ppi_formatted="partner: FOO")
    ex = adapt_row(row, et.get_target("bioreason_pro_test"))
    user = ex["prompt"]["user"]
    assert "Interaction partners" not in user
    assert "Organism: Homo sapiens" not in user
    assert "InterPro domains:" not in user
    assert "GO-GPT predictions:" not in user
    assert row["sequence"] in user


def test_adapt_row_includes_ppi_when_present_and_flagged():
    tgt = EvalTarget(name="t", hf_repo="r", has_ppi=True, has_interpro=True)
    ex = adapt_row(_test_data_row(ppi_formatted="And the following partners: FOO"), tgt)
    assert "Interaction partners:" in ex["prompt"]["user"]


def test_adapt_row_empty_assistant_turn_and_system():
    ex = adapt_row(_test_data_row(), et.get_target("bioreason_pro_test"))
    assert ex["prompt"]["assistant_reasoning"] == ""
    assert ex["prompt"]["assistant_answer"] == ""
    assert ex["prompt"]["system"] == "You are a protein function prediction assistant."
    assert ex["protein_id"] == "P123"


def test_adapt_row_truncates_sequence_to_cap():
    long_seq = "M" * (data.MAX_LENGTH_PROTEIN + 500)
    ex = adapt_row(_test_data_row(sequence=long_seq), et.get_target("bioreason_pro_test"))
    assert len(ex["sequence"]) == data.MAX_LENGTH_PROTEIN


def test_adapt_row_feeds_format_cafa5_without_keyerror():
    # The whole point: the protected formatter requires example["prompt"]; adapt_row provides it.
    ex = adapt_row(_test_data_row(), et.get_target("bioreason_pro_test"))
    formatted = data.format_cafa5_for_protein_llm(ex)   # must NOT KeyError
    user_turn = formatted["prompt"][0]
    assert user_turn["role"] == "user"
    content_types = [c["type"] for c in user_turn["content"]]
    assert content_types == ["protein", "go_graph", "text"]
    assert formatted["protein_sequences"] == [ex["sequence"]]


# --------------------------------------------------------------- registry / availability gate

def test_registry_lists_all_targets():
    assert set(et.TARGETS) == {"bioreason_pro_test", "cafa5", "cafa_no_knowledge"}


def test_get_target_bioreason_pro_test_available():
    t = et.get_target("bioreason_pro_test")
    assert t.hf_repo == "wanglab/bioreason-pro-test-data"
    assert t.hf_split == "test" and t.split_mode == "all" and t.has_ppi is False


def test_get_target_cafa5_refused_pending_access():
    with pytest.raises(RuntimeError, match="not available"):
        et.get_target("cafa5")


def test_get_target_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        et.get_target("does-not-exist")


# --------------------------------------------------------------- stream_eval_records (stubbed datasets)

def _install_fake_datasets(monkeypatch, rows):
    """Inject a fake `datasets` module so stream_eval_records' lazy `from datasets import load_dataset`
    resolves without the real library / network."""
    captured = {}

    def _load_dataset(repo, config=None, split=None, streaming=False, revision=None):
        captured["args"] = (repo, config, split, streaming, revision)
        return list(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = _load_dataset
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake)
    return captured


def test_stream_all_mode_yields_every_row_and_records_shape(monkeypatch):
    rows = [_test_data_row(protein_id="P1"), _test_data_row(protein_id="P2")]
    captured = _install_fake_datasets(monkeypatch, rows)
    recs = list(stream_eval_records(et.get_target("bioreason_pro_test")))
    # split_mode="all" → no in_split filter → all rows kept, streamed from the `test` split
    assert [r["protein_id"] for r in recs] == ["P1", "P2"]
    assert captured["args"] == (
        "wanglab/bioreason-pro-test-data",
        None,
        "test",
        True,
        "90e2fcbf2ad5c2e3da1753de140a627419f90fdb",
    )
    r0 = recs[0]
    assert set(r0) == {"protein_id", "user_turn", "sequence", "gt_terms"}
    assert r0["gt_terms"] == {"GO:0003674", "GO:0098772", "GO:0005575"}
    assert r0["user_turn"]["role"] == "user"


def test_stream_respects_limit(monkeypatch):
    rows = [_test_data_row(protein_id=f"P{i}") for i in range(5)]
    _install_fake_datasets(monkeypatch, rows)
    recs = list(stream_eval_records(et.get_target("bioreason_pro_test"), limit=2))
    assert len(recs) == 2


def test_stream_shards_partition_the_same_limited_prefix(monkeypatch):
    rows = [_test_data_row(protein_id=f"P{i}") for i in range(7)]
    _install_fake_datasets(monkeypatch, rows)
    even = list(
        stream_eval_records(
            et.get_target("bioreason_pro_test"), limit=5, shard_index=0, num_shards=2
        )
    )
    _install_fake_datasets(monkeypatch, rows)
    odd = list(
        stream_eval_records(
            et.get_target("bioreason_pro_test"), limit=5, shard_index=1, num_shards=2
        )
    )

    assert [r["protein_id"] for r in even] == ["P0", "P2", "P4"]
    assert [r["protein_id"] for r in odd] == ["P1", "P3"]
    assert {r["protein_id"] for r in even}.isdisjoint({r["protein_id"] for r in odd})


@pytest.mark.parametrize(
    ("shard_index", "num_shards"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_stream_rejects_invalid_shards(monkeypatch, shard_index, num_shards):
    _install_fake_datasets(monkeypatch, [_test_data_row()])
    with pytest.raises(ValueError):
        list(
            stream_eval_records(
                et.get_target("bioreason_pro_test"),
                shard_index=shard_index,
                num_shards=num_shards,
            )
        )


def test_stream_skips_rows_missing_id_or_sequence(monkeypatch):
    rows = [
        _test_data_row(protein_id="P1"),
        {"protein_id": "", "sequence": "M"},        # missing id
        {"protein_id": "P2", "sequence": ""},       # missing seq
        _test_data_row(protein_id="P3"),
    ]
    _install_fake_datasets(monkeypatch, rows)
    recs = list(stream_eval_records(et.get_target("bioreason_pro_test")))
    assert [r["protein_id"] for r in recs] == ["P1", "P3"]


# --------------------------------------------------------------- cafa_no_knowledge (plan.md Phase 1)

def test_cafa_no_knowledge_registered_local_and_available():
    tgt = et.get_target("cafa_no_knowledge")
    assert tgt.source == "local"
    assert tgt.split_mode == "all"


def test_cafa_no_knowledge_adapt_row_matches_bioreason_pro_test_prompt_shape(monkeypatch):
    """Both targets must produce byte-identical prompts for byte-identical rows: Phase 3 re-measures
    sft-checkpoint:v20/rl-checkpoint:v28 on this target, and those were trained on the
    data_contract prompt template, not eval_targets.base's generic one."""
    from eval_targets.bioreason_pro_test import TARGET as bpt_target

    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only_reasoned")
    row = {"protein_id": "Q1", "sequence": "MDYQRLLFLF", "go_mf": ["GO:0003674"],
           "go_bp": [], "go_cc": [], "interpro_formatted": "- IPR000001: some domain [1-50]"}
    cafa_ex = adapt_row(row, et.get_target("cafa_no_knowledge"))
    bpt_ex = adapt_row(row, bpt_target)
    assert cafa_ex["prompt"] == bpt_ex["prompt"]
    assert "IPR000001" in cafa_ex["prompt"]["user"]
    # ppi_formatted/subcellular_location are never collected for this target (module docstring) —
    # the reasoned template must still show its stable "not available" placeholder, not omit them.
    assert "Interaction partners (STRING):\nnot available" in cafa_ex["prompt"]["user"]
    assert "Subcellular location (UniProtKB):\nnot available" in cafa_ex["prompt"]["user"]
    assert cafa_ex["sequence"] == row["sequence"]
    assert parse_gt(row, et.get_target("cafa_no_knowledge")) == {"GO:0003674"}


def test_cafa_no_knowledge_adapt_row_sequence_only_for_unreasoned_variants(monkeypatch):
    monkeypatch.setenv("SENPAI_TARGET_VARIANT", "leaf_only")
    row = {"protein_id": "Q1", "sequence": "MDYQRLLFLF", "go_mf": ["GO:0003674"],
           "go_bp": [], "go_cc": [], "interpro_formatted": "- IPR000001: some domain [1-50]"}
    ex = adapt_row(row, et.get_target("cafa_no_knowledge"))
    assert "IPR000001" not in ex["prompt"]["user"]
    assert "not available" not in ex["prompt"]["user"]


def _write_local_jsonl(records, path=None):
    import json
    import tempfile

    if path is None:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        path = Path(handle.name)
    else:
        handle = path.open("w")
    with handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return path


def test_cafa_no_knowledge_stream_reads_local_jsonl(tmp_path):
    records = [
        {"protein_id": "Q1", "sequence": "MAAA", "go_mf": ["GO:0003674"], "go_bp": [], "go_cc": []},
        {"protein_id": "Q2", "sequence": "MBBB", "go_mf": [], "go_bp": ["GO:0009987"], "go_cc": []},
    ]
    jsonl_path = _write_local_jsonl(records, tmp_path / "dev.jsonl")
    tgt = dataclasses.replace(et.get_target("cafa_no_knowledge"), local_path=str(jsonl_path))
    recs = list(stream_eval_records(tgt))
    assert [r["protein_id"] for r in recs] == ["Q1", "Q2"]
    assert recs[0]["gt_terms"] == {"GO:0003674"}
    assert recs[1]["gt_terms"] == {"GO:0009987"}


def test_cafa_no_knowledge_stream_refuses_when_not_approved(tmp_path, monkeypatch):
    import bioreason_pro.license_policy as lp

    def _fake_load():
        return {"local_eval_targets": {}}  # no cafa_no_knowledge entry at all

    monkeypatch.setattr(lp, "load_approved_assets", _fake_load)
    jsonl_path = _write_local_jsonl([{"protein_id": "Q1", "sequence": "M"}], tmp_path / "dev.jsonl")
    tgt = dataclasses.replace(et.get_target("cafa_no_knowledge"), local_path=str(jsonl_path))
    with pytest.raises(Exception, match="not recorded"):
        list(stream_eval_records(tgt))
