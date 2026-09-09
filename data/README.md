# data/ — PROTECTED reference data (do not edit; not student-editable)

Reference files for IA-weighted F_max evaluation and GO ancestor propagation. These are **not
committed** (see `.gitignore`). Install the exact approved pair with:

```bash
python scripts/fetch_approved_reference_data.py
```

The script downloads the CC-BY-4.0 CAFA 5 evaluation bundle from Zenodo record `20186533`, verifies
the archive, verifies each extracted file, and refuses to overwrite a mismatched local file unless
the operator explicitly passes `--force`.

- `go-basic.obo` — Gene Ontology DAG (ancestor propagation + cafaeval namespaces).
- `IA.txt` — information-accretion weights (the "weighted" in weighted F_max).
- `eval_terms_no_knowledge_2025_03.tsv` — CAFA's no-knowledge evaluation targets; source for the
  Phase 1 temporal development set (plan.md, ADR-024/029).
- `known_t0.tsv` — CAFA's annotated-at-t0 proteins; used to verify what "no-knowledge" actually means
  for a given protein, not just trust the label.

`public_holdout_ids.txt` and `data_contract_audit.json` are small, committed audit artifacts. They
record the public holdout fingerprint, source revisions, raw overlaps, and the zero-overlap
operational split. Re-audit pinned remote revisions with:

```bash
uv run python scripts/audit_data_contract.py
```

`cafa_no_knowledge_ids.txt` and `cafa_no_knowledge_dev_set.jsonl` are the Phase 1 temporal dev set
(plan.md ADR-022/024/029) — also small, committed audit artifacts, built by
`scripts/build_cafa_no_knowledge_dev_set.py` (requires the two files above, plus
`data/public_holdout_ids.txt`, plus network access to UniProt). `cafa_no_knowledge_ids.txt` lists the
1,496 protein ids (CAFA's 1,717 no-knowledge targets, minus 213 in the sealed holdout, minus 8 found
in the training corpus — see ADR-029); the `.jsonl` file carries their sequences, GO ground truth, and
(once `scripts/fetch_interpro_annotations.py` has run) InterPro context.
`data/cafa_no_knowledge_corpus_disjointness.json` is the audit receipt from
`scripts/verify_cafa_no_knowledge_disjoint_from_corpus.py`, which re-streams the live training corpus
rather than trusting the cached exclusion list.

The gated `wanglab/cafa5` dataset remains disabled. The public
`wanglab/bioreason-pro-test-data` revision is the supported sealed holdout.
