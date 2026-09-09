#!/usr/bin/env python3
"""Verify the Phase 1 temporal dev set shares no protein with the training corpus, and record the result.

Streams the protein_id column of both approved training datasets (SFT and RL reasoning data, full
`train` split each) and checks against data/cafa_no_knowledge_ids.txt. This is separate from
scripts/build_cafa_no_knowledge_dev_set.py because it downloads ~600MB across two HF datasets — slow,
network-heavy, and not needed every time the dev set itself is rebuilt.

Writes data/cafa_no_knowledge_corpus_disjointness.json (committed): the checked dataset revisions,
corpus size, overlap count, and the date checked — the audit record method rule 6/12 asks for, rather
than an unverified comment asserting the two are disjoint by construction.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.dev_set import assert_disjoint_from_corpus  # noqa: E402
from bioreason_pro.license_policy import require_approved_dataset  # noqa: E402

IDS_PATH = ROOT / "data" / "cafa_no_knowledge_ids.txt"
RECEIPT_PATH = ROOT / "data" / "cafa_no_knowledge_corpus_disjointness.json"

CORPUS_DATASETS = (
    ("wanglab/bioreason-pro-sft-reasoning-data", "sft-training"),
    ("wanglab/bioreason-pro-rl-reasoning-data", "rl-training"),
)


def _stream_ids(repo_id: str, use: str, revision: str) -> set[str]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split="train", streaming=True, revision=revision)
    ids = set()
    for row in ds:
        pid = row.get("protein_id")
        if pid:
            ids.add(pid)
    return ids


def main() -> int:
    dev_ids = {line.strip() for line in IDS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}

    corpus_ids: set[str] = set()
    checked = []
    for repo_id, use in CORPUS_DATASETS:
        revision = require_approved_dataset(repo_id, use)
        ids = _stream_ids(repo_id, use, revision)
        print(f"[corpus] {repo_id}@{revision}: {len(ids)} unique protein ids", file=sys.stderr)
        corpus_ids |= ids
        checked.append({"repo_id": repo_id, "revision": revision, "protein_count": len(ids)})

    assert_disjoint_from_corpus(dev_ids, corpus_ids)  # raises DevSetError on any overlap

    receipt = {
        "dev_set_size": len(dev_ids),
        "corpus_datasets_checked": checked,
        "corpus_total_unique_ids": len(corpus_ids),
        "overlap_count": 0,
        "checked_on": date.today().isoformat(),
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[corpus] disjoint: 0 overlap out of {len(dev_ids)} dev-set ids vs {len(corpus_ids)} corpus ids")
    print(f"[corpus] wrote {RECEIPT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
