#!/usr/bin/env python3
"""Validate the pinned ESM2 config/tokenizer/BOS/EOS contract on an actual GPU model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch

    from bioreason_pro.protein_encoder import MAX_LENGTH_PROTEIN, create_protein_encoder

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the actual ESM2 contract check")
    encoder = create_protein_encoder(
        "facebook/esm2_t33_650M_UR50D", inference_mode=True, embedding_layer=33
    )
    encoder.model = encoder.model.to("cuda")
    lengths = (1, MAX_LENGTH_PROTEIN)
    encoded = encoder.encode_sequences(["M" * length for length in lengths], [0, 1], 2)
    observed = [int(value.shape[0]) for value in encoded]
    expected = [length + 2 for length in lengths]
    if observed != expected:
        raise RuntimeError(f"ESM2 token contract failed: expected={expected}, observed={observed}")
    report = {
        "model": encoder.model_name,
        "max_position_embeddings": int(encoder.model.config.max_position_embeddings),
        "special_tokens": int(encoder.tokenizer.num_special_tokens_to_add(pair=False)),
        "max_residues": encoder.max_residues,
        "hidden_size": encoder.embedding_dim,
        "observed_tokens": observed,
        "device": str(next(encoder.model.parameters()).device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ESM2-CONTRACT-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
