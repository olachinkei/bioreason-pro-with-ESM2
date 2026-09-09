"""eval_targets — the explicit, named held-out-dataset registry for sealed-test evaluation.

Each dataset is one module exporting a `TARGET` (an `EvalTarget`). `get_target(name)` resolves a
target by name and refuses ones that aren't available yet (e.g. `cafa5`, pending HF access). The
generation/scoring wiring (train.run_sealed_eval → eval.score_generations) is dataset-agnostic;
adding a dataset = adding a module here, nothing else.
"""

from __future__ import annotations

from eval_targets import bioreason_pro_test, cafa5, cafa_no_knowledge
from eval_targets.base import (
    EvalTarget,
    adapt_row,
    parse_gt,
    stream_eval_records,
)

TARGETS: dict[str, EvalTarget] = {
    t.name: t for t in (bioreason_pro_test.TARGET, cafa5.TARGET, cafa_no_knowledge.TARGET)
}

__all__ = [
    "EvalTarget",
    "TARGETS",
    "get_target",
    "adapt_row",
    "parse_gt",
    "stream_eval_records",
]


def get_target(name: str) -> EvalTarget:
    """Resolve an eval target by name. Raises if unknown or not yet available (access/schema pending)."""
    if name not in TARGETS:
        raise KeyError(f"unknown eval target {name!r}; known: {sorted(TARGETS)}")
    target = TARGETS[name]
    if not target.available:
        gate = " (HF access-gated)" if target.gated else ""
        raise RuntimeError(
            f"eval target {name!r} is not available yet{gate}; "
            f"resolve the TODO(pending access) items in eval_targets/{name}.py and set available=True"
        )
    return target
