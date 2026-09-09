#!/usr/bin/env python
"""Audit whether reasoning traces INVENT the evidence they cite (ADR-021).

The traces read as authoritative — `IPR005140 (eRF1/Pelota-like, N-terminal domain), residues 1-130`.
That is exactly what makes them dangerous if fabricated: a drug-discovery reviewer has no way to tell
an invented domain assignment from a real one. ADR-017 coupled supervision to the evidence in the
prompt on the theory that this prevents fabrication; this script tests the theory on real rollouts.

It pulls rollouts from Weave, which logs the prompt AND the trace for the same call, and checks every
InterPro id the trace asserts against the ids actually present in its own prompt.

    uv run python scripts/audit_reasoning_grounding.py --run fydegtkh

Grounding is necessary, not sufficient: a trace can cite every domain correctly and still reason
wrongly about function. No check here closes that gap — it needs a domain expert.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import sys

IPR = re.compile(r"IPR\d{6}")
RANGE = re.compile(r"residues?\s+(\d+)[–-](\d+)")


def fetch(project: str, run_id: str, limit: int) -> list[dict]:
    import weave

    client = weave.init(project)
    query = {"$expr": {"$eq": [{"$getField": "attributes.wandb_run_id"}, {"$literal": run_id}]}}
    out = []
    for call in client.get_calls(filter={"op_names": None}, query=query, limit=limit):
        i = call.inputs or {}
        if i.get("reasoning"):
            out.append({"sample_id": i.get("sample_id"), "step": i.get("rl_global_step"),
                        "prompt": str(i.get("prompt") or ""), "reasoning": str(i.get("reasoning") or ""),
                        "faithfulness": i.get("r_faithfulness")})
    return out


def audit(rows: list[dict]) -> dict:
    citing, ungrounded_rollouts, fracs, counts = 0, 0, [], []
    bad_ids: list[str] = []
    ranges = bad_ranges = 0
    for r in rows:
        prompt_ids = set(IPR.findall(r["prompt"]))
        trace_ids = set(IPR.findall(r["reasoning"]))
        if trace_ids:
            citing += 1
            counts.append(len(trace_ids))
            bad = trace_ids - prompt_ids
            fracs.append(len(bad) / len(trace_ids))
            if bad:
                ungrounded_rollouts += 1
                bad_ids += sorted(bad)
        for a, b in RANGE.findall(r["reasoning"]):
            ranges += 1
            # accept hyphen, en-dash and bracketed forms; InterPro renders ranges several ways
            if not any(f"{a}{sep}{b}" in r["prompt"] for sep in ("-", "–")):
                bad_ranges += 1
    return {
        "rollouts": len(rows), "citing_ipr": citing,
        "ids_per_rollout_mean": st.mean(counts) if counts else 0.0,
        "ids_per_rollout_max": max(counts) if counts else 0,
        "ungrounded_rollouts": ungrounded_rollouts,
        "ungrounded_fraction_mean": st.mean(fracs) if fracs else 0.0,
        "ungrounded_ids": sorted(set(bad_ids)),
        "ranges": ranges, "ranges_not_in_prompt": bad_ranges,
        "faithfulness_mean": st.mean([r["faithfulness"] for r in rows
                                      if isinstance(r.get("faithfulness"), (int, float))] or [0.0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="wandb-healthcare/bioreasonpro-senpai")
    ap.add_argument("--run", required=True, help="W&B run id whose rollouts to audit")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    rows = fetch(args.project, args.run, args.limit)
    if not rows:
        print(f"no rollouts with a reasoning trace found for run {args.run!r}", file=sys.stderr)
        return 1
    a = audit(rows)
    print(f"run {args.run}: {a['rollouts']} rollouts with a trace")
    print(f"  citing >=1 IPR id           : {a['citing_ipr']}")
    print(f"  IPR ids per rollout         : mean {a['ids_per_rollout_mean']:.1f}, "
          f"max {a['ids_per_rollout_max']}")
    print(f"  rollouts w/ ungrounded id   : {a['ungrounded_rollouts']} "
          f"({a['ungrounded_rollouts'] / max(a['citing_ipr'], 1) * 100:.2f}%)")
    print(f"  mean ungrounded fraction    : {a['ungrounded_fraction_mean'] * 100:.3f}%")
    if a["ungrounded_ids"]:
        print(f"  INVENTED IDS                : {a['ungrounded_ids'][:20]}")
    print(f"  residue ranges cited        : {a['ranges']}")
    print(f"  ranges not in prompt        : {a['ranges_not_in_prompt']} "
          f"({a['ranges_not_in_prompt'] / max(a['ranges'], 1) * 100:.1f}%)")
    print(f"  reward faithfulness (mean)  : {a['faithfulness_mean']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
