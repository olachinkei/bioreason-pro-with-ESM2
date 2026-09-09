#!/usr/bin/env python3
"""Reload an SFT checkpoint and reproduce one fused validation forward/generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    import torch

    import data
    import train

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is required for the SFT checkpoint reload smoke")

    started = time.monotonic()
    run_args = train._load_run_args(str(args.checkpoint))
    model, tokenizer = train.evaluate_checkpoint(str(args.checkpoint), run_args)
    device = next(model.parameters()).device
    example = data.synthetic_fixture_rows()[0]
    template = data.CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    batch = train._sft_forward_inputs(
        example, tokenizer, template, str(device), multimodal=True,
        max_seq_len=run_args.max_seq_len,
    )
    if batch is None:
        raise RuntimeError("fixture did not produce a supervised validation batch")
    with torch.no_grad():
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"reloaded checkpoint produced non-finite loss: {float(loss)}")

    formatted = data.format_cafa5_for_protein_llm(example)
    prompt = tokenizer.apply_chat_template(
        [formatted["prompt"][0]],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
    )
    prompt = data.expand_pad_tokens(prompt, example["sequence"])
    run_args.max_completion_length = args.max_new_tokens
    with torch.no_grad():
        response = train._fused_generate(
            model, tokenizer, prompt, example["sequence"], run_args
        )

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "model_name": run_args.model_name,
        "esm_model_name": run_args.esm_model_name,
        "protein_residues": len(example["sequence"]),
        "protein_tokens": len(example["sequence"]) + 2,
        "validation_loss": float(loss),
        "generated_tokens_requested": args.max_new_tokens,
        "generated_text": response,
        "elapsed_seconds": time.monotonic() - started,
    }
    if torch.cuda.is_available():
        report["gpu_peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    destination = args.checkpoint / "reload_verification.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PHASE3-RELOAD-OK: wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
