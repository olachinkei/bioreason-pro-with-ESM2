#!/usr/bin/env python3
"""Consume versioned SFT/RL Artifacts and compare them with one evaluation contract."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _evaluate_with_empty_prediction_fallback(evaluator, checkpoint, **kwargs):
    """Keep a bounded smoke comparison auditable when neither model emits a GO term."""
    try:
        return evaluator.evaluate(checkpoint, **kwargs)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-artifact", required=True)
    parser.add_argument("--rl-artifact", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("outputs/eval-artifacts"))
    parser.add_argument("--target", default="bioreason_pro_test")
    parser.add_argument("--subset-size", type=int, default=8)
    # Defaults to val. This comparison runs on EVERY GRPO invocation, so a "test" default means an
    # iterative search consumes the sealed holdout once per iteration — which plan.md forbids
    # ("Test data is never used for training, checkpoint selection, or hyperparameter selection").
    # Reading the sealed split is now an explicit, auditable choice for a one-off publication run.
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-name", default="")
    args = parser.parse_args()

    import torch
    import wandb

    import eval as evaluator
    import train
    from bioreason_pro.license_policy import validate_checkpoint_layout

    from bioreason_pro.wandb_meta import build_note, derived_tags

    job_type = "model-evaluation"
    config = {
        "sft_artifact": args.sft_artifact,
        "rl_artifact": args.rl_artifact,
        "target": args.target,
        "subset_size": args.subset_size,
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
            raise SystemExit("expected --sft-artifact stage=sft and --rl-artifact stage=rl")
        if rl_meta["sft_artifact"] != sft_ref:
            raise SystemExit(
                f"RL artifact was trained from {rl_meta['sft_artifact']}, not {sft_ref}"
            )

        if args.split == "test":
            print("[compare] WARNING: reading the SEALED holdout. This must be a one-off "
                  "publication measurement, never an input to selection or tuning.", flush=True)
        shared = {
            "split": args.split,
            "subset_size": args.subset_size,
            "target": args.target,
        }
        results = {}
        for label, checkpoint in (
            ("sft", sft_checkpoint),
            ("sft_to_grpo", rl_checkpoint),
        ):
            results[label] = _evaluate_with_empty_prediction_fallback(
                evaluator,
                checkpoint,
                out_dir=str(args.output.parent / f"{args.output.stem}-{label}"),
                **shared,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        report = {
            "schema_version": 2,
            "generation_and_scoring_contract": {
                **shared,
                "max_completion_length": 1024,
                "decoding": "greedy",
                "scorer": "cafaeval IA-weighted F_max",
            },
            "artifacts": {
                "sft": sft_ref,
                "sft_to_grpo": rl_ref,
                "rl_input_sft": rl_meta["sft_artifact"],
            },
            "metrics": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for label, metrics in results.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    run.log({f"comparison/{label}/{key}": value})
        comparison = wandb.Artifact(
            "bioreasonpro-phase4-comparison",
            type="evaluation",
            metadata={"sft_artifact": sft_ref, "rl_artifact": rl_ref},
        )
        comparison.add_file(str(args.output))
        run.log_artifact(comparison, aliases=["latest"]).wait()
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
