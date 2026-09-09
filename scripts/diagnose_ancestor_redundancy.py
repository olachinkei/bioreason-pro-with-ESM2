#!/usr/bin/env python3
"""How much of a model's generation budget goes to GO terms that propagation would add for free?

Reads stored rollout/eval generations, strips the terms implied by a more specific sibling, and
scores both prediction sets with the real metric. The scores should be identical; the difference in
set size is the share of the generation budget that bought nothing.

Input is either a W&B `run_table` artifact with `protein_id` and `completion` columns, or a local
JSON list of {"protein_id", "generated_response", "gt_terms"} records. Ground truth for the W&B
path is resolved from the approved rl-training dataset, so this never touches the sealed holdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as sealed_eval  # noqa: E402
from bioreason_pro.go_obo import load_go_ancestors  # noqa: E402
from bioreason_pro.rewards import extract_go_terms, strip_implied_ancestors  # noqa: E402

GO_ID = re.compile(r"GO:\d{7}")


def _records_from_wandb(artifact_refs: list[str], download_root: str) -> list[dict]:
    """Latest rollout per protein, joined to its approved rl-training ground truth."""
    import wandb

    import data

    api = wandb.Api()
    rows: list[dict] = []
    for index, artifact_ref in enumerate(artifact_refs):
        artifact = api.artifact(artifact_ref, type="run_table")
        directory = artifact.download(root=os.path.join(download_root, str(index)))
        path = glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)[0]
        table = json.loads(Path(path).read_text(encoding="utf-8"))
        rows += [dict(zip(table["columns"], row)) for row in table["data"]]

    latest: dict[str, dict] = {}
    for row in rows:
        pid = row["protein_id"]
        if pid not in latest or row.get("step", 0) >= latest[pid].get("step", 0):
            latest[pid] = row

    wanted = set(latest)
    truth: dict[str, set[str]] = {}
    for row in data.load_rl_prompts():
        pid = row["protein_id"]
        if pid in wanted and pid not in truth:
            terms: set[str] = set()
            for column in ("go_mf", "go_bp", "go_cc"):
                value = row.get(column) or []
                terms |= set(
                    value if isinstance(value, (list, set, tuple)) else str(value).split()
                )
            truth[pid] = {t for t in terms if GO_ID.fullmatch(t)}
        if len(truth) == len(wanted):
            break

    missing = wanted - set(truth)
    if missing:
        print(f"warning: no ground truth for {len(missing)} protein(s); excluded", file=sys.stderr)
    return [
        {
            "protein_id": pid,
            "generated_response": row["completion"],
            "gt_terms": truth[pid],
        }
        for pid, row in latest.items()
        if pid in truth
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--wandb-artifact",
        action="append",
        help="entity/project/run-<id>-<table>:vN (repeatable; versions are pooled)",
    )
    source.add_argument("--records", help="local JSON list of eval records")
    parser.add_argument("--out-dir", default="outputs/ancestor-redundancy")
    parser.add_argument("--download-root", default="outputs/rollout-tables")
    args = parser.parse_args()

    if args.wandb_artifact:
        records = _records_from_wandb(args.wandb_artifact, args.download_root)
    else:
        records = json.loads(Path(args.records).read_text(encoding="utf-8"))
        for record in records:
            record["gt_terms"] = set(record.get("gt_terms") or ())
    if not records:
        raise SystemExit("no records to analyse")

    ancestors = load_go_ancestors(sealed_eval.OBO)
    verbose, concise, kept_sizes, full_sizes = [], [], [], []
    for record in records:
        terms = extract_go_terms(record["generated_response"])
        kept = strip_implied_ancestors(terms, ancestors)
        full_sizes.append(len(terms))
        kept_sizes.append(len(kept))
        verbose.append({**record, "generated_response": " ".join(sorted(terms))})
        concise.append({**record, "generated_response": " ".join(sorted(kept))})

    mean_full = statistics.mean(full_sizes)
    mean_kept = statistics.mean(kept_sizes)
    print(f"records                     : {len(records)}")
    print(f"mean predicted terms        : {mean_full:.1f}")
    print(f"mean after stripping        : {mean_kept:.1f}")
    print(f"budget spent on free terms  : {1 - mean_kept / mean_full:.1%}" if mean_full else "")

    out = Path(args.out_dir)
    as_is = sealed_eval.score_generations(verbose, str(out / "verbose"), diagnostics=True)
    stripped = sealed_eval.score_generations(concise, str(out / "concise"), diagnostics=True)
    print(f"\n{'metric':<20} {'as-is':>10} {'stripped':>10} {'delta':>10}")
    # Aspects with no scored term are absent from cafaeval's frame, so report what came back.
    for key in sorted(k for k in as_is if k.startswith("weighted_fmax")):
        print(f"{key:<20} {as_is[key]:>10.6f} {stripped[key]:>10.6f} "
              f"{stripped[key] - as_is[key]:>+10.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
