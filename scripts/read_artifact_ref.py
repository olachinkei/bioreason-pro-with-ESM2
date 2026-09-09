#!/usr/bin/env python3
"""Print the immutable model reference from a local W&B artifact receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from bioreason_pro.license_policy import require_immutable_wandb_artifact_ref

    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    print(require_immutable_wandb_artifact_ref(str(payload.get("artifact_ref") or "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
