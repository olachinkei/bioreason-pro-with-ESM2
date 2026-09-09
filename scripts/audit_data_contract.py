#!/usr/bin/env python3
"""Audit pinned dataset IDs, write the sealed holdout IDs, and prove operational disjointness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.data_contract import assert_train_val_holdout_disjoint  # noqa: E402
from bioreason_pro.license_policy import load_approved_assets  # noqa: E402


def _urls(repo_id: str, spec: dict) -> list[str]:
    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/{spec['revision']}/{item['path']}"
        for item in spec["parquet_files"]
    ]


def _read_ids(connection, repo_id: str, spec: dict) -> list[str]:
    urls = ", ".join("'" + url.replace("'", "''") + "'" for url in _urls(repo_id, spec))
    rows = connection.execute(
        f"SELECT protein_id FROM read_parquet([{urls}]) WHERE protein_id IS NOT NULL"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _digest(ids: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write audited IDs and JSON report")
    args = parser.parse_args()

    import duckdb

    manifest = load_approved_assets()
    repos = {
        "sft_train": "wanglab/bioreason-pro-sft-reasoning-data",
        "rl_train": "wanglab/bioreason-pro-rl-reasoning-data",
        "public_holdout": "wanglab/bioreason-pro-test-data",
    }
    connection = duckdb.connect()
    raw = {
        name: _read_ids(connection, repo_id, manifest["datasets"][repo_id])
        for name, repo_id in repos.items()
    }
    unique = {name: set(ids) for name, ids in raw.items()}
    holdout = unique["public_holdout"]
    internal = (unique["sft_train"] | unique["rl_train"]) - holdout

    import data

    train = {pid for pid in internal if data.protein_split(pid) == "rl_train"}
    validation = {pid for pid in internal if data.protein_split(pid) == "val"}
    reserved_test = {pid for pid in internal if data.protein_split(pid) == "test"}
    assert_train_val_holdout_disjoint(train, validation, holdout)

    report = {
        "schema_version": 1,
        "audited_on": date.today().isoformat(),
        "contract_id": manifest["data_contract"]["id"],
        "sources": {
            name: {
                "repo_id": repo_id,
                "revision": manifest["datasets"][repo_id]["revision"],
                "rows": len(raw[name]),
                "unique_protein_ids": len(unique[name]),
                "duplicate_rows": len(raw[name]) - len(unique[name]),
                "sorted_id_sha256": _digest(unique[name]),
            }
            for name, repo_id in repos.items()
        },
        "raw_overlaps": {
            "sft_train_and_rl_train": len(unique["sft_train"] & unique["rl_train"]),
            "sft_train_and_public_holdout": len(unique["sft_train"] & holdout),
            "rl_train_and_public_holdout": len(unique["rl_train"] & holdout),
            "combined_training_sources_and_public_holdout": len(
                (unique["sft_train"] | unique["rl_train"]) & holdout
            ),
        },
        "operational_splits_after_holdout_exclusion": {
            "rl_train": len(train),
            "validation": len(validation),
            "reserved_internal_test": len(reserved_test),
            "public_holdout": len(holdout),
            "pairwise_overlap": 0,
        },
    }
    expected = manifest["data_contract"]["public_holdout_ids"]
    if len(holdout) != expected["count"] or _digest(holdout) != expected["sha256"]:
        raise SystemExit("public holdout IDs do not match approved_assets.json")

    ids_path = ROOT / expected["path"]
    report_path = ROOT / "data" / "data_contract_audit.json"
    if args.write:
        ids_path.write_text("\n".join(sorted(holdout)) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {ids_path} and {report_path}")
    else:
        if not ids_path.is_file() or not report_path.is_file():
            raise SystemExit("audited artifacts are missing; rerun with --write")
        recorded = json.loads(report_path.read_text(encoding="utf-8"))
        recorded.pop("audited_on", None)
        report.pop("audited_on", None)
        if recorded != report:
            raise SystemExit("data contract audit report is stale; rerun with --write")
        print("Data contract audit artifacts match the pinned remote revisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
