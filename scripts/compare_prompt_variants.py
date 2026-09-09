#!/usr/bin/env python3
"""A/B one checkpoint's val eval across prompt variants on a single GPU.

Tests whether telling the model to emit only the most specific GO terms converts the wasted budget
measured by `scripts/diagnose_ancestor_redundancy.py` into score. Learning-free: the same weights
are decoded twice, once per variant, on the same deterministic val subset.

Validation only — `split="val"` is the in-loop steering split. The sealed holdout is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="immutable model artifact entity/project/name:vN")
    parser.add_argument("--artifact-root", default="outputs/prompt-variant/models")
    parser.add_argument("--output", default="outputs/prompt-variant/comparison.json")
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--variants", default="baseline,specific_only")
    parser.add_argument(
        "--completion-lengths",
        default=None,
        help="comma-separated max_completion_length values to sweep instead of prompt variants; "
             "the model fills whatever budget it is given, so this is a precision knob",
    )
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    lengths = None
    if args.completion_lengths:
        lengths = [int(v) for v in args.completion_lengths.split(",") if v.strip()]
        if len(lengths) < 2 or any(v < 1 for v in lengths):
            raise SystemExit("--completion-lengths needs >=2 positive values")
        # Sweeping the budget holds the prompt fixed, so only the first variant is used.
        variants = variants[:1]
    elif len(variants) < 2:
        raise SystemExit("--variants needs at least two comma-separated names to compare")

    from eval_targets.base import PROMPT_VARIANTS

    unknown = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; expected from {sorted(PROMPT_VARIANTS)}")
    if args.subset_size < 1:
        raise SystemExit("--subset-size must be positive")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("prompt-variant evaluation requires a CUDA GPU")

    import wandb
    import weave

    import train
    from bioreason_pro.license_policy import validate_checkpoint_layout
    from bioreason_pro.weave_eval import WEAVE_PROJECT

    # Weave's on-disk response cache uses sqlite, which SLURM's TMPDIR / NFS $HOME cannot open.
    os.environ.setdefault("WEAVE_USE_SERVER_CACHE", "false")
    os.environ.setdefault("WEAVE_SERVER_CACHE_DIR", "/tmp/weave_cache")

    from bioreason_pro.wandb_meta import build_note, derived_tags

    job_type = "prompt-variant-val-eval"
    config = {
        "artifact": args.artifact,
        "variants": variants,
        "split": "val",
        "subset_size": args.subset_size,
        "max_completion_length": args.max_completion_length,
        "source_revision": os.environ.get("BIOREASON_SOURCE_REVISION", ""),
    }
    run = wandb.init(
        entity="wandb-healthcare",
        project="bioreasonpro-senpai",
        name=args.wandb_name or None,
        job_type=job_type,
        config=config,
        tags=derived_tags(job_type, config),
        notes=build_note(job_type, config, extra=f"lengths={args.max_completion_length}"),
    )
    # Without this the Evaluation is dropped inside Weave's background executor and the run looks
    # like it published. train.py does the same for the training path.
    weave.init(WEAVE_PROJECT)
    try:
        checkpoint, reference = train.use_model_artifact(run, args.artifact, args.artifact_root)
        meta = validate_checkpoint_layout(checkpoint)
        run.summary["checkpoint_artifact_ref"] = reference
        run.summary["checkpoint_stage"] = meta["stage"]

        run_args = train._load_run_args(
            checkpoint,
            {
                "eval_subset_size": args.subset_size,
                "max_completion_length": args.max_completion_length,
                "wandb_name": args.wandb_name or "prompt-variant",
            },
        )
        model, tokenizer = train.evaluate_checkpoint(checkpoint, run_args)

        results = {}
        arms = (
            [(f"len{v}", variants[0], v) for v in lengths]
            if lengths
            else [(v, v, args.max_completion_length) for v in variants]
        )
        for label, variant, length in arms:
            os.environ["SENPAI_EVAL_PROMPT_VARIANT"] = variant
            run_args.max_completion_length = length
            metrics = train._generation_eval(
                model, tokenizer, run_args, multimodal=True, split="val",
                publish_weave_eval=True,
            )
            results[label] = {**dict(metrics), "max_completion_length": length,
                              "prompt_variant": variant}
            run.log({f"arm/{label}/{k}": v for k, v in metrics.items()})
            n_aspects = metrics.get("weighted_fmax_n_aspects")
            warn = ""
            if n_aspects is not None and n_aspects < 3:
                warn = (f"  !! NOT COMPARABLE: mean over {int(n_aspects)} aspect(s), not 3 — the "
                        f"model predicted nothing for the missing one(s)")
            print(f"[{label}] max_completion_length={length} "
                  f"weighted_fmax={metrics.get('weighted_fmax')}{warn}", flush=True)
        os.environ.pop("SENPAI_EVAL_PROMPT_VARIANT", None)

        baseline = next(iter(results))
        for variant in list(results)[1:]:
            delta = results[variant].get("weighted_fmax", 0.0) - results[baseline].get(
                "weighted_fmax", 0.0
            )
            run.summary[f"delta/{variant}_minus_{baseline}"] = delta
            print(f"delta {variant} - {baseline} = {delta:+.6f}", flush=True)

        payload = {
            "schema_version": 1,
            "artifact": reference,
            "split": "val",
            "subset_size": args.subset_size,
            "max_completion_length": args.max_completion_length,
            "variants": results,
            "source_revision": os.environ.get("BIOREASON_SOURCE_REVISION", ""),
            "wandb_run": f"{run.entity}/{run.project}/{run.id}",
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
