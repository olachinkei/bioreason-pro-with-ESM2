#!/usr/bin/env python
"""Re-derive every number in docs/ROLLOUT_ANALYSIS.md from the stored RL rollout tables.

Each RL run logs `rl/rollout_samples` as a W&B Table at a handful of optimizer steps. One table is
one GRPO group: `num_generations` completions of a **single** protein. That shape matters and is the
easiest thing to get wrong here — see `--paired` below.

    uv run python scripts/analyze_rollouts.py              # every section
    uv run python scripts/analyze_rollouts.py --section redundancy

Redundancy is measured against the real ontology (`data/go-basic.obo`) with the same ancestor closure
the scorer uses, so "redundant" means what it means in ADR-004: a term that is a proper ancestor of
another term the same rollout emitted, and therefore already implied by it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = "wandb-healthcare/bioreasonpro-senpai"
OBO = ROOT / "data" / "go-basic.obo"

# run id -> label. Kept explicit so the population being described is auditable, not "whatever the
# API returned today".
CONDITIONS = {
    "pfsrs9mp": "aspect_mean / leaf_only (recommended, seed 0)",
    "5k353896": "aspect_mean / leaf_only (seed 1)",
    "40cvaeze": "union / leaf_only (seed 1)",
    "49wvo70n": "union / leaf_only (50 steps)",
    "41fsrvnj": "union / full_closure (shipped baseline)",
    "vnd8beks": "aspect_mean_specific / leaf_only (null result)",
    "sonpumcq": "aspect_mean + beta=0.04 (harmful)",
    "c1tfuk6v": "aspect_mean gen16 (single-seed max)",
}
UNION_ARMS = ["40cvaeze", "49wvo70n"]
ASPECT_ARMS = ["pfsrs9mp", "5k353896"]
ASPECTS = ("MF", "BP", "CC")
GO_RE = re.compile(r"GO:\d{7}")


def fetch(cache: Path | None) -> dict:
    if cache and cache.exists():
        return json.loads(cache.read_text())
    import wandb

    api = wandb.Api()
    out = {}
    for rid, label in CONDITIONS.items():
        run = api.run(f"{PROJECT}/{rid}")
        rows = []
        for art in (a for a in run.logged_artifacts() if a.type == "run_table"):
            path = Path(art.download()) / "rl" / "rollout_samples.table.json"
            if not path.exists():
                continue
            table = json.loads(path.read_text())
            rows += [dict(zip(table["columns"], r)) for r in table["data"]]
        out[rid] = {"label": label, "rows": rows}
    if cache:
        cache.write_text(json.dumps(out))
    return out


def terms_by_aspect(completion: str) -> dict[str, list[str]]:
    out = {}
    for aspect in ASPECTS:
        line = re.search(rf"^{aspect}:\s*(.*)$", completion, re.M)
        out[aspect] = GO_RE.findall(line.group(1)) if line else []
    return out


def section_shape(data: dict) -> None:
    print("== population ==")
    total = 0
    for rid, blob in data.items():
        steps = sorted({r["step"] for r in blob["rows"]})
        proteins = {r["step"]: {x["protein_id"] for x in blob["rows"] if x["step"] == r["step"]}
                    for r in blob["rows"]}
        one_per_step = all(len(v) == 1 for v in proteins.values())
        total += len(blob["rows"])
        print(f"  {rid}  {len(blob['rows']):3d} rollouts  steps={steps}  "
              f"one-protein-per-step={one_per_step}  {blob['label']}")
    print(f"  TOTAL {total} rollouts across {len(data)} conditions")
    print("  NOTE: one protein per step means cross-step comparisons are confounded by protein\n"
          "        identity. Use --section paired for the comparison that controls for it.")


def section_degenerate(data: dict) -> None:
    """Findings that hold across the whole population regardless of condition."""
    rows = [r for b in data.values() for r in b["rows"]]
    empty = sum(1 for r in rows if re.search(r"<think>\s*</think>", r["completion"]))
    truncated = sum(1 for r in rows if re.search(r"GO:\d{1,6}$", r["completion"].rstrip()))
    print("\n== whole-population properties ==")
    print(f"  empty <think> block : {empty}/{len(rows)} ({empty / len(rows) * 100:.1f}%)")
    print(f"  cut mid-identifier  : {truncated}/{len(rows)} ({truncated / len(rows) * 100:.1f}%)")

    from collections import Counter
    aux = Counter(round(r["reward"] - r["fmax"], 9) for r in rows)
    print("\n== reward minus F_max (what the auxiliary terms contributed) ==")
    for value, n in aux.most_common():
        print(f"  {value:+.3f} : {n:4d} rollouts ({n / len(rows) * 100:.1f}%)")
    print("  A component with no within-group variance contributes nothing to a GRPO advantage,\n"
          "  which is centred within the group. See ADR-013.")


def section_redundancy(data: dict) -> None:
    sys.path.insert(0, str(ROOT))
    from bioreason_pro.go_obo import load_go_ancestors

    ancestors = load_go_ancestors(OBO)
    print("\n== ancestor redundancy at the final logged step (ADR-004's mechanism) ==")
    print(f"  {'condition':46s} {'MF':>5} {'BP':>5} {'CC':>5} {'total':>6} {'redundant':>10}")
    for rid, blob in data.items():
        last_step = max(r["step"] for r in blob["rows"])
        rows = [r for r in blob["rows"] if r["step"] == last_step]
        parsed = [terms_by_aspect(r["completion"]) for r in rows]
        counts = {a: st.mean(len(p[a]) for p in parsed) for a in ASPECTS}
        fractions = []
        for p in parsed:
            emitted = set(sum(p.values(), []))
            if not emitted:
                continue
            redundant = sum(1 for t in emitted
                            if any(t != u and t in ancestors.get(u, set()) for u in emitted))
            fractions.append(redundant / len(emitted))
        share = st.mean(fractions) * 100 if fractions else 0.0
        print(f"  {blob['label'][:45]:46s} {counts['MF']:5.1f} {counts['BP']:5.1f} "
              f"{counts['CC']:5.1f} {sum(counts.values()):6.1f} {share:9.1f}%")


def section_paired(data: dict) -> None:
    """union vs aspect_mean with protein identity controlled.

    Both arms consume the same data order, so at a given step they score the *same* protein. Pairing
    on (step, protein) removes the confound that makes a naive across-step trend meaningless.
    """
    def aggregate(rids):
        cells: dict[tuple, list] = {}
        for rid in rids:
            for r in data[rid]["rows"]:
                cells.setdefault((r["step"], r["protein_id"]), []).append(
                    terms_by_aspect(r["completion"]))
        return {k: {a: st.mean(len(p[a]) for p in v) for a in ASPECTS} for k, v in cells.items()}

    union, aspect = aggregate(UNION_ARMS), aggregate(ASPECT_ARMS)
    keys = sorted(set(union) & set(aspect))
    print("\n== union vs aspect_mean, paired on (step, protein) — ADR-005's mechanism ==")
    print(f"  {'step':>5} {'protein':12s} {'union MF/BP/CC':>22} {'aspect_mean MF/BP/CC':>24} "
          f"{'CC share U':>11} {'CC share A':>11}")
    shares_u, shares_a = [], []
    for key in keys:
        u, a = union[key], aspect[key]
        tu, ta = sum(u.values()), sum(a.values())
        cu, ca = (u["CC"] / tu * 100 if tu else 0), (a["CC"] / ta * 100 if ta else 0)
        shares_u.append(cu)
        shares_a.append(ca)
        print(f"  {key[0]:5d} {key[1]:12s} {u['MF']:6.2f}/{u['BP']:.2f}/{u['CC']:.2f}"
              f"{'':8}{a['MF']:6.2f}/{a['BP']:.2f}/{a['CC']:.2f}{'':8}{cu:8.0f}% {ca:10.0f}%")
    deltas = [x - y for x, y in zip(shares_u, shares_a)]
    se = st.stdev(deltas) / sqrt(len(deltas))
    print(f"\n  mean CC share: union {st.mean(shares_u):.1f}%  aspect_mean {st.mean(shares_a):.1f}%")
    print(f"  paired difference {st.mean(deltas):+.1f} points, t = {st.mean(deltas) / se:.2f} "
          f"(df = {len(deltas) - 1})")
    print("  Step 1 is identical by construction: same checkpoint, same prompt, before either\n"
          "  reward has applied a gradient.")


def section_signal(data: dict) -> None:
    print("\n== within-group reward spread (the only thing GRPO can learn from) ==")
    for rid, blob in data.items():
        by_step: dict[int, list[float]] = {}
        for r in blob["rows"]:
            by_step.setdefault(r["step"], []).append(r["reward"])
        sds = {s: (st.stdev(v) if len(v) > 1 else 0.0) for s, v in sorted(by_step.items())}
        dead = sum(1 for v in sds.values() if v < 1e-9)
        print(f"  {blob['label'][:45]:46s} {[f'{v:.3f}' for v in sds.values()]}"
              + (f"  <-- {dead} zero-variance step(s)" if dead else ""))


SECTIONS = {
    "shape": section_shape,
    "degenerate": section_degenerate,
    "redundancy": section_redundancy,
    "paired": section_paired,
    "signal": section_signal,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=sorted(SECTIONS), action="append")
    ap.add_argument("--cache", type=Path, help="reuse/store the fetched rollouts as JSON")
    args = ap.parse_args()

    data = fetch(args.cache)
    for name in args.section or list(SECTIONS):
        SECTIONS[name](data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
