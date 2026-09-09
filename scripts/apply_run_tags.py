#!/usr/bin/env python
"""Apply the W&B run-tag vocabulary from a declarative table. Dry-run by default.

The rules and the reasoning live in docs/WANDB_TAGS.md. In short: a tag exists only to carry a
judgement (recommended / superseded / null-result / not-comparable / broken-config) or to repair a condition
the run config never recorded. Everything else should be read from the config or from artifact
lineage.

    uv run python scripts/apply_run_tags.py            # print the diff
    uv run python scripts/apply_run_tags.py --apply    # write it

Idempotent: running twice changes nothing the second time. Provenance and stage are derived — via
`bioreason_pro.wandb_meta`, the same module every `wandb.init()` call now uses live — so new runs
carry them from the moment they start; this script only needs to backfill runs older than that
change, plus every judgement and repair, which still need a human's table entry.
"""

from __future__ import annotations

import argparse
import sys

from bioreason_pro.wandb_meta import JOB_TYPE_STAGE, provenance as _provenance

PROJECT = "wandb-healthcare/bioreasonpro-senpai"

# `best` is deprecated everywhere. It conflated "highest number seen" with "the recipe we recommend",
# and in this project those differ: 0.30980 was the luckiest of three seeds and quoting it was
# selection on noise. Use `recommended` (established, replicated) or `single-seed-max` (a number, not
# a result). The tagger strips `best` wherever it survives.
DEPRECATED_TAGS = {"best"}

# --- Explicit assignments -----------------------------------------------------------------------
# Judgements and backfills only. Key = run id.
#
# Backfill provenance: the reward variant was recovered from the Slurm job's stdout on the cluster
# (`grep "reward variant" runtime_logs/phase4-grpo-<job>.out`), because reward_variant did not enter
# the W&B config until PR #142. The job number is recorded so the recovery is re-checkable.

RUNS: dict[str, dict] = {
    # ---- Phase 3/4 originals (the shipped baseline) ----
    "t97inugd": {"tags": ["tv-full-closure", "superseded"], "note": "job 59, SFT v2, shipped baseline"},
    "41fsrvnj": {"tags": ["rw-union", "tv-full-closure", "superseded"], "note": "job 64, RL v2"},
    "hhn45qgc": {"tags": ["sealed"], "note": "job 74, Phase 5 full holdout, 8630 proteins"},
    # ---- Phase 7 SFT arms ----
    "qwtttrzy": {"tags": ["tv-leaf-only", "superseded"], "remove": ["recommended"], "note": "job 627, SFT v4; the original single-seed supervision arm (+0.0715). Replaced by the Phase 8 three-seed set at HEAD, which measured +0.06228 at t=12.1."},
    "s76awz5e": {"tags": ["tv-leaf-mf-only", "superseded"], "note": "job 631, rejected"},
    "82lp21lw": {"tags": ["tv-leaf-only"], "note": "job 655-era, esm_layer 24; better SFT, worse RL start"},
    "yt8atf30": {"tags": ["tv-leaf-only", "not-comparable"], "note": "esm_layer 28, in-loop only, never swept at its own optimum"},
    "hc7hwu1s": {"tags": ["tv-leaf-only", "not-comparable"], "note": "esm_layer 30, in-loop only, never swept at its own optimum"},
    "tnsszsuj": {"tags": ["tv-leaf-only", "smoke"], "note": "val=0, plumbing"},
    "18lxbn8r": {"tags": ["smoke"], "note": "val=0, plumbing"},
    "ksp581yx": {"tags": ["smoke"], "note": "val=0, plumbing"},
    # ---- Phase 7 RL arms: reward variant backfilled from cluster stdout ----
    "dzqp3mx2": {"tags": ["rw-union", "tv-leaf-only", "superseded"], "note": "job 637, 100 steps"},
    "49wvo70n": {"tags": ["rw-union", "tv-leaf-only", "superseded"], "note": "job 641, 50 steps, union default"},
    "ulhgxsk1": {"tags": ["rw-union", "tv-leaf-only", "superseded"], "note": "25 steps"},
    "eqxogtwl": {"tags": ["rw-union", "tv-leaf-only", "superseded"], "note": "25 steps"},
    "pfsrs9mp": {"tags": ["rw-aspect-mean", "tv-leaf-only", "seed0", "n3", "recommended"], "note": "job 645, aspect_mean seed 0; the established recipe (n3 mean 0.27594 +/- 0.0069, 6.3 sigma over union)"},
    "jqdo6tzr": {"tags": ["rw-aspect-mean", "tv-leaf-only", "superseded"], "note": "job 649, 100 steps, collapses"},
    "vqqw8vpu": {"tags": ["rw-aspect-mean", "tv-leaf-only", "superseded"], "note": "job 651, 100 steps + beta 0.04"},
    "1wb2ihm5": {"tags": ["rw-aspect-mean", "tv-leaf-only", "superseded"], "note": "esm_layer 24 + RL: the better SFT is the worse RL start"},
    "c1tfuk6v": {"tags": ["rw-aspect-mean", "tv-leaf-only", "seed0", "n3", "single-seed-max"], "note": "job 685, gen16 seed 0; 0.30980 is the highest single val-256 number and is NOT the result (n3 mean 0.29327 +/- 0.0145)"},
    "k2wzke7c": {"tags": ["rw-aspect-mean", "tv-leaf-only", "superseded"], "note": "gen32, saturated at 4x wall-clock"},
    "4gx67x0m": {"tags": ["rw-aspect-mean", "tv-leaf-only"], "note": "job 689, lora_r 64, tied peak / broader optimum"},
    "9ygnae6t": {"tags": ["rw-aspect-mean", "tv-leaf-only", "n3"], "note": "job 692, gen16 seed 1"},
    "2tlj7noq": {"tags": ["rw-aspect-mean", "tv-leaf-only", "n3"], "note": "job 693, gen16 seed 2"},
    "5k353896": {"tags": ["rw-aspect-mean", "tv-leaf-only", "n3", "recommended"], "note": "job 700, gen8 seed 1 — BACKFILL: config says rv=None"},
    "fihoh5x0": {"tags": ["rw-aspect-mean", "tv-leaf-only", "n3", "recommended"], "note": "job 701, gen8 seed 2 — BACKFILL: config says rv=None"},
    "40cvaeze": {"tags": ["rw-union", "tv-leaf-only", "n3"], "note": "job 702, union seed 1 — BACKFILL: config says rv=None"},
    "sfbujw2l": {"tags": ["rw-union", "tv-leaf-only", "n3"], "note": "job 703, union seed 2 — BACKFILL: config says rv=None"},
    "vnd8beks": {"tags": ["tv-leaf-only", "n3", "null-result"], "note": "job 708, specificity seed 0; mechanism live, metric flat"},
    "vnb5uq1v": {"tags": ["tv-leaf-only", "n3", "null-result"], "note": "job 709, specificity seed 1"},
    "fas65b1r": {"tags": ["tv-leaf-only", "n3", "null-result"], "note": "job 710, specificity seed 2"},
    "sonpumcq": {"tags": ["tv-leaf-only", "n3", "superseded"], "note": "job 714, beta 0.04 at 50 steps seed 0; anchor destroys the RL gain"},
    "f7vbs2js": {"tags": ["tv-leaf-only", "n3", "superseded"], "note": "job 715, beta 0.04 seed 1"},
    "er7jl4pg": {"tags": ["tv-leaf-only", "n3", "superseded"], "note": "job 716, beta 0.04 seed 2"},
    "bcw22fxl": {"tags": ["broken-config"], "note": "job 691, --rl_algo gspo: the multimodal loop has no IS ratio, so this was a GRPO duplicate. Cancelled at step 9."},
    "y1f8u66m": {"tags": ["broken-config"], "note": "job 677, PHASE4_ESM_LAYER did not exist; fail-closed check caught the mismatch"},
    # ---- Phase 8: the supervision leg replicated, 3 seeds per arm, all six at commit 4148839 ----
    # No tv-* tags here: these runs record `target_variant`, and `train_seed`, in their own config,
    # so a tag would be a second source of truth. This is the first set where that is true.
    "dipscney": {"tags": ["smoke"], "note": "job 724, train_seed 1 knob smoke; loss 2.1613 vs 1.6033 at seed 2, manifests identical"},
    "o6qfg2qu": {"tags": ["smoke"], "note": "job 725, train_seed 2 knob smoke; the pair proved PHASE3_TRAIN_SEED moves the RNG and not the split"},
    "0h6v6khx": {"tags": ["n3", "recommended"], "note": "job 727, leaf_only train_seed 0; SFT of record (n3 mean 0.24889 +/- 0.0040 at len 48)"},
    "9a9m5lgt": {"tags": ["n3", "recommended"], "note": "job 729, leaf_only train_seed 1"},
    "ig0bcru0": {"tags": ["n3", "recommended"], "note": "job 731, leaf_only train_seed 2"},
    "v3325kgl": {"tags": ["n3"], "note": "job 726, full_closure train_seed 0; control arm (n3 mean 0.18661 +/- 0.0080)"},
    "qdxlmzgq": {"tags": ["n3"], "note": "job 728, full_closure train_seed 1"},
    "d6jzsw9b": {"tags": ["n3"], "note": "job 730, full_closure train_seed 2"},
    # ---- Evaluation sweeps: judgements only; the artifact is on the lineage edge ----
    "lr4o0yo2": {"tags": ["val", "base"], "remove": ["smoke"], "note": "the as-is reference point (0.11227 at the shipped 1024 budget); every +N% is measured against this. NOT a smoke run."},
    "ez0vno3e": {"tags": ["val", "single-seed-max"], "note": "gen16 seed 0 sweep, 0.30980 — the number that was over-quoted; not a result"},
    "2rnmf2k3": {"tags": ["val", "n3"], "note": "gen16 seed 1, 0.28709"},
    "oloqsqqi": {"tags": ["val", "n3"], "note": "gen16 seed 2, 0.28292"},
    "hz0o9s5y": {"tags": ["val", "n3"], "note": "gen8 aspect_mean seed 1, 0.26844"},
    "7wxzfdlc": {"tags": ["val", "n3"], "note": "gen8 aspect_mean seed 2, 0.28188"},
    "til5nqpx": {"tags": ["val", "n3"], "note": "gen8 union seed 1, 0.24966"},
    "40pxhn5f": {"tags": ["val", "n3"], "note": "gen8 union seed 2, 0.24754"},
    "xte1ss9b": {"tags": ["val", "n3", "null-result"], "note": "specificity seed 0"},
    "c3iyolqj": {"tags": ["val", "n3", "null-result"], "note": "specificity seed 1"},
    "q2x33xqp": {"tags": ["val", "n3", "null-result"], "note": "specificity seed 2"},
    "7i3ycn8m": {"tags": ["val", "n3"], "note": "beta 0.04 seed 0, 0.24066"},
    "aa92iu0c": {"tags": ["val", "n3"], "note": "beta 0.04 seed 1, 0.24639"},
    "qqat1a5n": {"tags": ["val", "n3"], "note": "beta 0.04 seed 2, 0.24896"},
    # ---- Phase 9: the authorised sealed read (2026-08-14). Three seeds, budget 64, 8,630 proteins.
    # Authorised explicitly by the owner; option A = seal the existing recommended RL checkpoints,
    # so the spread below is RL-seed variance with the SFT held fixed at sft-checkpoint:v4.
    "gtklbm01": {"tags": ["sealed", "smoke"], "note": "job 778, 4-protein plumbing smoke at budget 64; n_aspects=1 by construction, never a result"},
    "53xz4re3": {"tags": ["sealed", "n3", "recommended"], "note": "job 779, rl-checkpoint:v6 seed 0 — sealed 0.17422 (SFT arm 0.17136)"},
    "pp1clrgx": {"tags": ["sealed", "n3", "recommended"], "note": "job 780, rl-checkpoint:v16 seed 1 — sealed 0.17265 (SFT arm 0.17005)"},
    "wglc5ivr": {"tags": ["sealed", "n3", "recommended"], "note": "job 781, rl-checkpoint:v17 seed 2 — sealed 0.18774 (SFT arm 0.17063); the seed that carries the RL delta"},
    # ---- Phase 11: the reasoning objective (ADR-014) and why it is blocked (ADR-015, ADR-016) ----
    "porblb9u": {"tags": ["rl", "broken-config"], "note": "job 836, aspect_mean_reasoned smoke at budget 512: substance_sd=0, the reward term was inert because supervision manufactures an empty <think>. Nothing was tested."},
    "7o853421": {"tags": ["smoke"], "note": "job 915, mask_empty_think knob smoke; span matched under the real tokenizer (the run is fail-closed on zero matches)"},
    "jzvrzbk5": {"tags": ["null-result"], "note": "job 916, sft-checkpoint:v18, leaf_only + mask_empty_think: reasoning_nonempty still 0 AND in-loop fmax 0.0778 vs 0.21718. Mechanism confirmed live, metric moved the wrong way. ADR-016."},
    # ---- Phase 11 (ADR-017): reasoning supervision from the dataset's traces ----
    "31o06dm6": {"tags": ["new-series"], "note": "job 975, sft-checkpoint:v20, leaf_only_reasoned; prompt carries InterPro/STRING/location"},
    "o2wffvz0": {"tags": ["val", "new-series"], "note": "reasoned sweep 64/1024; len64 has n_aspects=2 and is unrankable"},
    "1t36godm": {"tags": ["val", "new-series"], "note": "reasoned sweep 64/1536 — peak 0.18900, 100% non-empty traces, faithfulness 0.991"},
    "vzfz0uqz": {"tags": ["val", "new-series"], "note": "reasoned sweep 64/2048"},
    "fydegtkh": {"tags": ["rl", "new-series"], "note": "job 1009, rl-checkpoint:v28, aspect_mean_reasoned 50 steps @1536; val -0.0165 vs its SFT (ADR-019). 385 rollouts audited: 0 ungrounded IPR ids (ADR-021)."},
    "fp8jsvf6": {"tags": ["sealed", "new-series"], "note": "job 1040, SECOND authorised sealed read: SFT v20 0.32674, RL v28 0.33289, paired +0.00615 - below one RL seed sd, and OPPOSITE in sign to val (ADR-020). Recipe postdates the first sealed read."},
    "of3zwkpd": {"tags": ["val", "not-comparable"], "note": "len16 arm scored MF only: 0.47668 is a one-aspect mean"},
    "oyfkjdw7": {"tags": ["val", "not-comparable"], "note": "len32 arm, two-aspect mean 0.30663 once reported as a +21.8% win"},
    "dysq9dk9": {"tags": ["val", "not-comparable"], "note": "len24 bracket extension, two-aspect mean"},
    "1q78d27w": {"tags": ["val", "not-comparable"], "note": "len32 bracket extension, two-aspect mean"},
    "b63k7jpi": {"tags": ["val", "not-comparable"], "note": "len32 bracket extension, two-aspect mean"},
    "qjsbur5v": {"tags": ["val", "superseded"], "remove": ["recommended"], "note": "leaf_only SFT sweep, 0.24701, single seed; superseded by the Phase 8 three-seed mean 0.24889 +/- 0.0040"},
    # ---- Phase 8 sweeps: every peak bracketed on both sides, n_aspects == 3 at all 24 points ----
    "9crwtepo": {"tags": ["val", "n3", "recommended"], "note": "job 775, leaf_only seed 0, peak 0.25137 @ 48"},
    "c4jcvl1g": {"tags": ["val", "n3", "recommended"], "note": "job 776, leaf_only seed 1, peak 0.25103 @ 48"},
    "2knfjbu5": {"tags": ["val", "n3", "recommended"], "note": "job 777, leaf_only seed 2, peak 0.24428 @ 48"},
    "mf66erzp": {"tags": ["val", "n3"], "note": "job 772, full_closure seed 0, peak 0.18866 @ 192"},
    "r5qkxmvj": {"tags": ["val", "n3"], "note": "job 773, full_closure seed 1, peak 0.19337 @ 256"},
    "vqmp0ogx": {"tags": ["val", "n3"], "note": "job 774, full_closure seed 2, peak 0.17780 @ 192"},
    "uxmolz6h": {"tags": ["val", "superseded"], "remove": ["rl"], "note": "union RL 50 steps, 0.25184; held a stale `best` — beaten by aspect_mean at 6.3 sigma"},
    "w6iqic5t": {"tags": ["val", "superseded"], "note": "full_closure RL at its optimum, 0.20009"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write tags (default is dry-run)")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    runs = list(api.runs(args.project))
    changed = kept = 0

    for run in runs:
        try:
            config = dict(run.config)
        except Exception:
            config = {}
        desired = {_provenance(config)}
        stage = JOB_TYPE_STAGE.get(run.job_type or "")
        if stage:
            desired.add(stage)
        entry = RUNS.get(run.id)
        if entry:
            desired.update(entry["tags"])
        # An eval run must carry a split; default to val, since sealed access is deliberate.
        if "eval" in desired and not ({"val", "sealed"} & desired):
            desired.add("val")

        current = set(run.tags or [])
        strip = DEPRECATED_TAGS | set((entry or {}).get("remove", []))
        merged = (current | desired) - strip
        if merged == current:
            kept += 1
            continue
        changed += 1
        added = sorted(merged - current)
        removed = sorted(current - merged)
        label = (entry or {}).get("note", "")
        delta = ("+" + ",".join(added) if added else "") + ("  -" + ",".join(removed) if removed else "")
        print(f"{run.id}  {delta}" + (f"   # {label}" if label else ""))
        if args.apply:
            run.tags = sorted(merged)
            run.update()

    verb = "applied" if args.apply else "would change"
    print(f"\n{verb}: {changed} run(s); already correct: {kept}; total: {len(runs)}")
    if not args.apply:
        print("dry run — pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
