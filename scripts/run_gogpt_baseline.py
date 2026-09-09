#!/usr/bin/env python3
"""GO-GPT baseline (plan.md Phase 5 / ADR-028): score the paper's own zero-LLM GO-term predictor on
the Phase 1 `cafa_no_knowledge` dev set, using the same real `cafaeval` scorer
(`eval.score_generations`) every other arm in this ledger is scored by (ADR-030: never the reward's
F1 proxy). This measures what BioReason-Pro's reasoning layer adds over a much cheaper discrete
annotator -- the paper's own results say that margin is thin.

Model code is vendored from bowang-lab/BioReason-Pro (MIT) at third_party/gogpt/ rather than
reimplemented -- see third_party/gogpt/NOTICE.md for why. Not adopted as a BioReason-Pro prompt
field: owner ruling 2026-08-23, baseline measurement only for now.

This is NOT a reproduction of the paper's own reported 0.65/0.70: that number is measured on the
paper's own 8,630-protein set with an unconfirmed decoding protocol (plan.md ADR-028 / PAPER.md).
This measures the same released checkpoint on THIS project's 1,496-protein temporal dev set, using
GOGPTPredictor.predict()'s own default beam search (beam_size=5) -- a comparable, honestly-labelled
number, not an attempted exact match.

Usage:
    uv run python scripts/run_gogpt_baseline.py                # full 1,496-protein dev set
    uv run python scripts/run_gogpt_baseline.py --limit 3       # smoke test, no cafaeval needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOGPT_SRC = ROOT / "third_party" / "gogpt" / "src"

from bioreason_pro.baselines import as_generated_response  # noqa: E402
from bioreason_pro.data_contract import parse_go_terms  # noqa: E402
from bioreason_pro.license_policy import require_approved_model  # noqa: E402

ASPECTS = ("go_mf", "go_bp", "go_cc")


def _load_dev_set() -> list[dict]:
    path = ROOT / "data" / "cafa_no_knowledge_dev_set.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_gt(row: dict) -> set[str]:
    gt: set[str] = set()
    for aspect in ASPECTS:
        gt |= parse_go_terms(row.get(aspect))
    return gt


def union_predicted_terms(predictions: dict[str, list[str]]) -> set[str]:
    """Flatten GOGPTPredictor.predict()'s {"MF": [...], "BP": [...], "CC": [...]} into one term set.

    eval.score_generations does its own per-aspect scoring and ancestor propagation from the raw
    GO ids in generated_response (ADR-030's discipline: the real scorer, not a hand-rolled one), so
    this just needs to hand it every id GO-GPT predicted, aspect-tagging is not this function's job.
    """
    terms: set[str] = set()
    for aspect_terms in predictions.values():
        terms.update(aspect_terms)
    return terms


def load_gogpt_predictor(model_id: str = "wanglab/gogpt"):
    """Fail closed on GO-GPT's own weights AND the encoder its config declares before loading either.

    GOGPTPredictor.from_pretrained reads embed_model_path out of the checkpoint's own downloaded
    config.yaml and loads whatever it says via transformers.AutoModel -- so if that config ever
    changes to point at an unapproved encoder, this check catches it before the vendored code acts
    on it, the same fail-closed discipline every other model/dataset load in this project follows.
    """
    from huggingface_hub import hf_hub_download
    import yaml

    from bioreason_pro.license_policy import approved_model_revision

    normalized = require_approved_model(model_id, "go_decoder")
    revision = approved_model_revision(normalized, "go_decoder")
    config_path = hf_hub_download(repo_id=normalized, filename="config.yaml", revision=revision)
    embed_model_path = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["model"][
        "embed_model_path"
    ]
    require_approved_model(embed_model_path, "protein")
    # NOTE: GOGPTPredictor/GOGPT (vendored, unmodified) call AutoTokenizer/AutoModel.from_pretrained
    # on the encoder with no revision argument at all -- there is no hook to force
    # approved_assets.json's pinned encoder revision without editing vendored code. The name-approval
    # check above is what actually matters for the fail-closed property (never load an arbitrary,
    # unapproved encoder); accepting HF's own "main" resolution for a long-published, stable Meta
    # checkpoint is a known, accepted gap, not a silent one.

    if str(GOGPT_SRC) not in sys.path:
        sys.path.insert(0, str(GOGPT_SRC))
    from gogpt.inference import GOGPTPredictor  # vendored -- see third_party/gogpt/NOTICE.md

    return GOGPTPredictor.from_pretrained(normalized, revision=revision)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="score only the first N dev-set proteins (smoke test)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "gogpt_baseline"))
    args = parser.parse_args()

    rows = _load_dev_set()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[gogpt-baseline] scoring {len(rows)} proteins", file=sys.stderr)

    predictor = load_gogpt_predictor()

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        predictions = predictor.predict(sequence=row["sequence"], organism=row.get("organism") or "Unknown")
        records.append({
            "protein_id": row["protein_id"],
            "generated_response": as_generated_response(union_predicted_terms(predictions)),
            "gt_terms": _row_gt(row),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"[gogpt-baseline] predicted {i + 1}/{len(rows)}", file=sys.stderr)

    import eval as sealed_eval

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = sealed_eval.score_generations(records, str(out_dir), diagnostics=True)
    print(f"[gogpt-baseline] weighted_fmax={metrics['weighted_fmax']:.5f} "
          f"n_aspects={metrics.get('weighted_fmax_n_aspects')} "
          f"coverage={metrics.get('diag/coverage'):.3f}", file=sys.stderr)

    (out_dir / "results.json").write_text(
        json.dumps({"n_proteins": len(rows), "metrics": metrics}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[gogpt-baseline] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
