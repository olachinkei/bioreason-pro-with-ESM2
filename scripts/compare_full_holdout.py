#!/usr/bin/env python3
"""Distributed, lineage-aware SFT versus GRPO comparison on the sealed public holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FULL_HOLDOUT_SIZE = 8_630


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_files(paths: list[Path], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {len(missing)} shard/receipt files")
        time.sleep(2)


def _job_type_for_target(target: str) -> str:
    """'full-holdout-evaluation' means the sealed holdout specifically. Any other target (e.g.
    plan.md Phase 3's cafa_no_knowledge dev set) gets its own job_type, so bioreason_pro.wandb_meta's
    note never claims "sealed holdout" for a run that never touched it (docs/WANDB_TAGS.md: a
    `sealed` label appearing where it isn't warranted is an incident, not a curiosity)."""
    return "full-holdout-evaluation" if target == "bioreason_pro_test" else "dev-set-evaluation"


ALL_LABELS = ("sft", "sft_to_grpo")


def _parse_labels(raw: str) -> tuple[str, ...]:
    """Comma-separated --labels string -> validated subset of ALL_LABELS, order-preserving.

    Lets SFT and RL be submitted as two independent jobs (e.g. one node each, in parallel) instead
    of one job evaluating both checkpoints sequentially."""
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not labels or any(label not in ALL_LABELS for label in labels):
        raise ValueError(f"--labels must be a comma-separated subset of {ALL_LABELS}, got {raw!r}")
    return labels


def _expected_count(subset_size: int | None, full_size: int) -> int:
    if subset_size is not None:
        if subset_size < 1:
            raise ValueError("--subset-size must be a positive integer")
        return subset_size
    if full_size < 1:
        raise ValueError("--expected-full-size must be a positive integer")
    return full_size


def _merge_and_validate_shards(
    shard_root: Path,
    labels: tuple[str, ...],
    world_size: int,
    expected_count: int,
) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    reference_ground_truth: dict[str, set[str]] | None = None
    for label in labels:
        by_id: dict[str, dict] = {}
        for rank in range(world_size):
            path = shard_root / f"{label}-rank-{rank:02d}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                protein_id = row["protein_id"]
                if protein_id in by_id:
                    raise ValueError(f"duplicate protein_id in {label} shards: {protein_id}")
                by_id[protein_id] = {
                    "protein_id": protein_id,
                    "generated_response": row["generated_response"],
                    "gt_terms": set(row["gt_terms"]),
                }
        if len(by_id) != expected_count:
            raise ValueError(
                f"{label} coverage is {len(by_id)}, expected exactly {expected_count} proteins"
            )
        ground_truth = {protein_id: row["gt_terms"] for protein_id, row in by_id.items()}
        if reference_ground_truth is None:
            reference_ground_truth = ground_truth
        elif ground_truth != reference_ground_truth:
            raise ValueError(f"{label} protein IDs or ground-truth labels differ from the SFT set")
        merged[label] = [by_id[protein_id] for protein_id in sorted(by_id)]
    return merged


def _score_with_empty_prediction_fallback(evaluator, records, out_dir: str) -> dict:
    try:
        return evaluator.score_generations(records, out_dir)
    except KeyError as exc:
        if "cafa_eval returned no 'f_w'/'f' best-scores frame" not in str(exc):
            raise
        return {
            "weighted_fmax": 0.0,
            "weighted_fmax_mf": 0.0,
            "weighted_fmax_bp": 0.0,
            "weighted_fmax_cc": 0.0,
            "scoring_status": "empty_predictions",
            "scoring_error": str(exc),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-artifact", required=True)
    parser.add_argument("--rl-artifact", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--target", default="bioreason_pro_test")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--expected-full-size", type=int, default=FULL_HOLDOUT_SIZE)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--coordination-timeout-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-name", default="")
    parser.add_argument(
        "--labels",
        default="sft,sft_to_grpo",
        help="comma-separated subset of {sft, sft_to_grpo} to generate/score this run -- lets SFT "
             "and RL be submitted as two independent jobs (e.g. one node each) instead of one job "
             "evaluating both checkpoints sequentially. Both artifacts are still downloaded and "
             "lineage-checked regardless, so a single-label run still confirms the RL checkpoint's "
             "parent SFT artifact matches.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_count = _expected_count(args.subset_size, args.expected_full_size)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if args.max_completion_length < 1:
        raise ValueError("--max-completion-length must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a CUDA GPU")
    torch.cuda.set_device(local_rank)

    import eval as evaluator
    import train
    from bioreason_pro.data_contract import require_active_variant_matches
    from bioreason_pro.license_policy import validate_checkpoint_layout

    args.shard_root.mkdir(parents=True, exist_ok=True)
    receipt_path = args.shard_root / "artifact-inputs.json"
    run = None
    if rank == 0:
        import wandb

        from bioreason_pro.wandb_meta import build_note, derived_tags

        job_type = _job_type_for_target(args.target)
        config = {
            "sft_artifact": args.sft_artifact,
            "rl_artifact": args.rl_artifact,
            "target": args.target,
            "subset_size": args.subset_size,
            "expected_count": expected_count,
            "max_completion_length": args.max_completion_length,
            "world_size": world_size,
            "source_revision": os.environ.get("BIOREASON_SOURCE_REVISION", ""),
        }
        run = wandb.init(
            entity="wandb-healthcare",
            project="bioreasonpro-senpai",
            name=args.wandb_name or None,
            job_type=job_type,
            config=config,
            tags=derived_tags(job_type, config),
            notes=build_note(job_type, config),
        )
        try:
            sft_checkpoint, sft_ref = train.use_model_artifact(
                run, args.sft_artifact, str(args.artifact_root / "sft")
            )
            rl_checkpoint, rl_ref = train.use_model_artifact(
                run, args.rl_artifact, str(args.artifact_root / "rl")
            )
            sft_meta = validate_checkpoint_layout(sft_checkpoint)
            rl_meta = validate_checkpoint_layout(rl_checkpoint)
            if sft_meta["stage"] != "sft" or rl_meta["stage"] != "rl":
                raise ValueError("expected immutable stage=sft and stage=rl model artifacts")
            if rl_meta["sft_artifact"] != sft_ref:
                raise ValueError(f"RL parent {rl_meta['sft_artifact']} does not match {sft_ref}")
            # target_variant isn't in run_args.json (data_contract reads it from an env var, never a
            # checkpoint field), only in the producing run's own W&B config -- check it there. Missed
            # once already: this whole sweep ran on the wrong (context-free) prompt template for a
            # reasoned checkpoint because nothing caught SENPAI_TARGET_VARIANT being left unset
            # (plan.md ADR-031's retracted first Phase 3 read).
            sft_variant = wandb.Api().artifact(args.sft_artifact).logged_by().config.get("target_variant")
            rl_variant = wandb.Api().artifact(args.rl_artifact).logged_by().config.get("target_variant")
            if sft_variant is not None and rl_variant is not None and sft_variant != rl_variant:
                raise ValueError(
                    f"SFT target_variant={sft_variant!r} != RL target_variant={rl_variant!r} -- "
                    "refusing an inconsistent paired comparison"
                )
            require_active_variant_matches(sft_variant, checkpoint_label=f"sft_artifact={args.sft_artifact}")
            require_active_variant_matches(rl_variant, checkpoint_label=f"rl_artifact={args.rl_artifact}")
            _atomic_write_json(
                receipt_path,
                {
                    "sft_checkpoint": str(Path(sft_checkpoint).resolve()),
                    "sft_artifact": sft_ref,
                    "rl_checkpoint": str(Path(rl_checkpoint).resolve()),
                    "rl_artifact": rl_ref,
                    "rl_input_sft": rl_meta["sft_artifact"],
                },
            )
        except Exception:
            run.finish(exit_code=1)
            raise
    else:
        _wait_for_files([receipt_path], args.coordination_timeout_seconds)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    labels = _parse_labels(args.labels)
    checkpoints = {
        "sft": receipt["sft_checkpoint"],
        "sft_to_grpo": receipt["rl_checkpoint"],
    }
    for label in labels:
        records = train.run_sealed_eval(
            checkpoints[label],
            target=args.target,
            split="test",
            subset_size=args.subset_size,
            max_completion_length=args.max_completion_length,
            shard_index=rank,
            num_shards=world_size,
        )
        serializable = [
            {**row, "gt_terms": sorted(row["gt_terms"])}
            for row in records
        ]
        _atomic_write_json(args.shard_root / f"{label}-rank-{rank:02d}.json", serializable)
        del records, serializable
        gc.collect()
        torch.cuda.empty_cache()

    if rank != 0:
        return 0

    try:
        shard_paths = [
            args.shard_root / f"{label}-rank-{shard_rank:02d}.json"
            for label in labels
            for shard_rank in range(world_size)
        ]
        _wait_for_files(shard_paths, args.coordination_timeout_seconds)
        merged = _merge_and_validate_shards(
            args.shard_root, labels, world_size, expected_count
        )
        metrics = {
            label: _score_with_empty_prediction_fallback(
                evaluator,
                merged[label],
                str(args.output.parent / f"{args.output.stem}-{label}"),
            )
            for label in labels
        }

        # Per-protein inspection in Weave: score_generations above already computed the metric of
        # record (attached as attributes, never recomputed here). This mirrors train._generation_eval's
        # publish_weave_eval=True path -- this script previously did not call it at all, so the sealed
        # holdout's own per-protein generations were never traced anywhere (plan.md ADR-041/042).
        from bioreason_pro import go_obo, weave_eval

        obo_ancestors = go_obo.load_go_ancestors(evaluator.OBO)
        ia = go_obo.load_ia_weights(evaluator.IA)
        weave_evaluations = {}
        for label in labels:
            url = weave_eval.publish_evaluation(
                merged[label],
                metrics[label],
                obo_ancestors=obo_ancestors,
                ia=ia,
                name=f"eval-full-holdout-{label}-{args.wandb_name or run.id}",
                attributes={
                    "target": args.target,
                    "evaluation_scope": "subset" if args.subset_size is not None else "full_holdout",
                    "label": label,
                    "max_completion_length": args.max_completion_length,
                },
            )
            if url:
                weave_evaluations[label] = url
                print(f"WEAVE-EVALUATION[{label}]: {url}", flush=True)

        protein_ids = [row["protein_id"] for row in merged[labels[0]]]
        report = {
            "schema_version": 1,
            "evaluation_scope": "subset" if args.subset_size is not None else "full_holdout",
            "generation_and_scoring_contract": {
                "target": args.target,
                "split": "test",
                "labels": list(labels),
                "subset_size": args.subset_size,
                "evaluated_proteins": expected_count,
                "protein_ids_sha256": hashlib.sha256(
                    ("\n".join(protein_ids) + "\n").encode("utf-8")
                ).hexdigest(),
                "max_completion_length": args.max_completion_length,
                "decoding": "greedy",
                "scorer": "cafaeval IA-weighted F_max",
                "num_shards": world_size,
            },
            "source_revision": os.environ.get("BIOREASON_SOURCE_REVISION", ""),
            "wandb_run": f"{run.entity}/{run.project}/{run.id}",
            "artifacts": {
                "sft": receipt["sft_artifact"],
                "sft_to_grpo": receipt["rl_artifact"],
                "rl_input_sft": receipt["rl_input_sft"],
            },
            "metrics": metrics,
            "weave_evaluations": weave_evaluations,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for label, values in metrics.items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    run.log({f"comparison/{label}/{key}": value})
        run.summary["evaluated_proteins"] = expected_count
        run.summary["evaluation_scope"] = report["evaluation_scope"]

        import wandb

        comparison = wandb.Artifact(
            "bioreasonpro-phase5-full-holdout",
            type="evaluation",
            metadata={
                "sft_artifact": receipt["sft_artifact"],
                "rl_artifact": receipt["rl_artifact"],
                "evaluation_scope": report["evaluation_scope"],
                "evaluated_proteins": expected_count,
                "labels": list(labels),
            },
        )
        comparison.add_file(str(args.output))
        alias = "full-holdout" if args.subset_size is None else "smoke"
        logged = run.log_artifact(comparison, aliases=["latest", alias])
        logged.wait()
        artifact_ref = train._qualified_artifact_ref(logged, run)
        run.summary["comparison_artifact_ref"] = artifact_ref
        _atomic_write_json(
            args.output.parent / "wandb_artifact.json",
            {
                "schema_version": 1,
                "artifact_ref": artifact_ref,
                "artifact_type": "evaluation",
                "producer_run": report["wandb_run"],
                "sft_artifact": receipt["sft_artifact"],
                "rl_artifact": receipt["rl_artifact"],
                "evaluation_scope": report["evaluation_scope"],
                "evaluated_proteins": expected_count,
            },
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"WANDB-ARTIFACT: {artifact_ref}", flush=True)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
