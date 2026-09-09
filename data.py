"""data.py — PROTECTED. HF streaming loaders, CAFA5 chat formatting, multimodal collation, and the
split/label contract. Changing this breaks comparability and integrity.

Vendored faithfully from public bioreason2 (dataset/cafa5/format.py, collate.py, processor.py) with
the correctness gotchas the source encodes preserved (see inline notes). The pure functions
(formatting, pad-token expansion, label masking, split predicate) are unit-tested without HF/GPU;
the streaming loaders and the jinja render are the only HF-touching parts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bioreason_pro.special_tokens import GO_GRAPH_PAD_TOKEN, PROTEIN_PAD_TOKEN

SPLITS = ("rl_train", "val", "test")
CHAT_TEMPLATE_PATH = Path(__file__).parent / "bioreason_pro" / "qwen3_4b_chat_template.jinja2"

# Pad-token expansion magic numbers — MUST match the model's embedding insertion (bioreason_pro.model):
MAX_LENGTH_PROTEIN = 1024   # ESM2 config: 1026 positions minus BOS/EOS
NUM_GO_TOKENS = 200         # fixed reduced GO embeddings per aspect


# --------------------------------------------------------------- splits / leakage

def protein_split(protein_id: str, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1) -> str:
    """Deterministically assign a protein to exactly one split by hashing its ID (stable across runs)."""
    h = hashlib.sha256(f"{seed}:{protein_id}".encode()).hexdigest()
    frac = int(h[:16], 16) / float(1 << 64)
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "rl_train"


def in_split(protein_id: str, split: str, seed: int = 0) -> bool:
    """Filter to an internal split after excluding every sealed public-holdout protein."""
    from bioreason_pro.data_contract import is_public_holdout

    if is_public_holdout(protein_id):
        return False
    return protein_split(protein_id, seed) == split


def assert_disjoint_splits(rl_train_ids, val_ids, test_ids) -> None:
    """Fail fast if any protein ID appears in more than one split (leakage guard)."""
    rl, va, te = set(rl_train_ids), set(val_ids), set(test_ids)
    overlaps = {"rl_train∩val": rl & va, "rl_train∩test": rl & te, "val∩test": va & te}
    bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
    if bad:
        raise AssertionError(f"split leakage detected (protein IDs in >1 split): {bad}")


def deterministic_val_subset(val_ids, size: int = 256, seed: int = 0) -> list[str]:
    """Fixed, reproducible val subset for in-loop eval (sorted by hash, not insertion order)."""
    return sorted(val_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())[:size]


# --------------------------------------------------------------- formatting (pure, vendored verbatim)

def format_cafa5_for_protein_llm(example: dict[str, Any]) -> dict[str, Any]:
    """Format one CAFA5 row into the exact chat shape the jinja template + collator consume.

    Gotchas preserved: system prompt is FOLDED into the user text (no system message);
    reasoning_content is a SIBLING of the assistant content list; protein/go_graph are placeholder
    content items (ONE pad token each — expanded later); input col is 'sequence' (singular).
    """
    p = example["prompt"]
    answer = p["assistant_answer"].strip()
    return {
        "prompt": [
            {"role": "user", "content": [
                {"type": "protein", "text": None},
                {"type": "go_graph", "text": None},
                {"type": "text", "text": f"{p['system'].strip()}\n\n{p['user'].strip()}"},
            ]},
            {"role": "assistant",
             "reasoning_content": p["assistant_reasoning"].strip(),
             "content": [{"type": "text", "text": answer}]},
        ],
        "protein_sequences": [example["sequence"]],
        "structure_path": example.get("structure_path"),
        "go_aspect": example.get("go_aspect"),
        "answer": answer,
        "ground_truth_go_terms": example.get("ground_truth_go_terms", ""),
    }


def expand_pad_tokens(text: str, sequence: str, max_length_protein: int = MAX_LENGTH_PROTEIN,
                      num_go_tokens: int = NUM_GO_TOKENS) -> str:
    """Expand the single <|protein_pad|>/<|go_graph_pad|> placeholder into the right token counts.

    protein → min(len(sequence), max_length_protein) + 2 copies (the +2 = ESM BOS/EOS);
    go_graph → exactly num_go_tokens copies (independent of protein). Uses a temp marker so a
    naive replace(pad, pad*n) doesn't infinitely re-match (upstream processor pattern).
    """
    n_prot = min(len(sequence), max_length_protein) + 2
    text = text.replace(PROTEIN_PAD_TOKEN, "\x00P" * n_prot, 1).replace("\x00P", PROTEIN_PAD_TOKEN)
    text = text.replace(GO_GRAPH_PAD_TOKEN, "\x00G" * num_go_tokens, 1).replace("\x00G", GO_GRAPH_PAD_TOKEN)
    return text


def _find_subseq(seq: list[int], sub: list[int], start: int = 0) -> int:
    """Index of the first occurrence of `sub` in `seq` at/after `start`, else -1."""
    if not sub:
        return -1
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def mask_assistant_labels(input_ids: list[int], assistant_marker: list[int], end_marker: list[int],
                          pad_id: int) -> list[int]:
    """Assistant-only SFT labels: -100 everywhere except tokens strictly BETWEEN each
    '<|im_start|>assistant\\n' marker (exclusive) and the next '<|im_end|>' (exclusive); pad → -100.

    `assistant_marker`/`end_marker` are the token-id encodings of those strings (add_special_tokens
    =False). Pure/testable with hand-built id sequences.
    """
    labels = [-100] * len(input_ids)
    pos = 0
    while True:
        s = _find_subseq(input_ids, assistant_marker, pos)
        if s == -1:
            break
        content_start = s + len(assistant_marker)
        e = _find_subseq(input_ids, end_marker, content_start)
        if e == -1:
            break  # trailing generation-prompt header with no following <|im_end|> → no labels
        for i in range(content_start, e):
            labels[i] = input_ids[i]
        pos = e + len(end_marker)
    for i, tok in enumerate(input_ids):
        if tok == pad_id:
            labels[i] = -100
    return labels


# --------------------------------------------------------------- HF-touching (guarded)

def render_chat(example: dict[str, Any], tokenizer) -> str:
    """Render a formatted example to one flat string via the vendored qwen3 template.

    Do NOT pass enable_thinking (keeps the real <think>reasoning</think>). Expands pad tokens after
    templating. Requires the two special tokens already added to the tokenizer vocab.
    """
    formatted = format_cafa5_for_protein_llm(example)
    template = CHAT_TEMPLATE_PATH.read_text()
    text = tokenizer.apply_chat_template(
        formatted["prompt"], tokenize=False, add_generation_prompt=True, chat_template=template)
    return expand_pad_tokens(text, formatted["protein_sequences"][0])


# --------------------------------------------------------------- text-only (Stage 0/1)

def format_text_prompt(example: dict[str, Any]) -> str:
    """Plain-text prompt for the text-only stages (no protein/GO placeholders). system + user."""
    p = example["prompt"]
    return f"{p['system'].strip()}\n\n{p['user'].strip()}"


# A tiny, checked-in smoke set (no HF/gated access) using the same sequence/GO-only contract.
SYNTHETIC_FIXTURE_PATH = (
    Path(__file__).parent / "tests" / "fixtures" / "approved_training_rows.jsonl"
)


def synthetic_fixture_rows() -> list[dict[str, Any]]:
    """Load the checked-in offline fixture through the same minimal field adapter."""
    from bioreason_pro.data_contract import adapt_synthetic_fixture_row

    return [
        adapt_synthetic_fixture_row(json.loads(line))
        for line in SYNTHETIC_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tiny_rl_smoke_dataset():
    """A tiny text-only GRPO dataset using the checked-in approved synthetic fixture."""
    from datasets import Dataset

    rows = []
    for approved in synthetic_fixture_rows():
        rows.append({
            "prompt": format_text_prompt(approved),
            "protein_id": approved["protein_id"],
            "go_mf": approved["go_mf"],
            "go_bp": approved["go_bp"],
            "go_cc": approved["go_cc"],
        })
    return Dataset.from_list(rows)


def load_rl_prompts_text(n: int = 64, seed: int = 0):
    """Text-only RL prompts from wanglab/bioreason-pro-rl-reasoning-data (rl_train split), first n
    rows as an in-memory datasets.Dataset for GRPOTrainer. Keeps go_* for the reward."""
    from datasets import Dataset

    stream = _load_stream("wanglab/bioreason-pro-rl-reasoning-data")
    rows = []
    for r in stream:
        if not in_split(r["protein_id"], "rl_train", seed):
            continue
        from bioreason_pro.data_contract import adapt_approved_training_row

        approved = adapt_approved_training_row(
            r, repo_id="wanglab/bioreason-pro-rl-reasoning-data", use="rl-training"
        )
        rows.append({"prompt": format_text_prompt(approved), "protein_id": approved["protein_id"],
                     "go_mf": approved["go_mf"], "go_bp": approved["go_bp"],
                     "go_cc": approved["go_cc"]})
        if len(rows) >= n:
            break
    return Dataset.from_list(rows)


# --------------------------------------------------------------- streaming loaders

def _load_stream(repo: str, config: str | None = None):
    from bioreason_pro.license_policy import approved_dataset_spec
    from datasets import load_dataset

    use = "holdout-evaluation" if repo == "wanglab/bioreason-pro-test-data" else (
        "sft-training" if repo == "wanglab/bioreason-pro-sft-reasoning-data" else "rl-training"
    )
    spec = approved_dataset_spec(repo, use)
    return load_dataset(
        repo, config, split=spec["hf_split"], streaming=True, revision=spec["revision"]
    )


def load_sft_dataset(subset_frac: float = 1.0, streaming: bool = True, seed: int = 0):
    """Stream wanglab/bioreason-pro-sft-reasoning-data (~124k), formatted for SFT. Never materializes
    CAFA-5. `subset_frac`<1 takes a deterministic head slice for smoke runs.

    Rows carry the nested `prompt` dict + `sequence` + `protein_id`; we .map(format_...) and drop
    the val/test proteins so the SFT signal never overlaps the sealed eval splits.
    """
    ds = _load_stream("wanglab/bioreason-pro-sft-reasoning-data")
    ds = ds.filter(lambda r: in_split(r["protein_id"], "rl_train", seed))  # exclude val/test proteins

    columns = ds.column_names
    ds = ds.map(
        lambda r: _adapt_or_mark_unusable(
            r, repo_id="wanglab/bioreason-pro-sft-reasoning-data", use="sft-training"
        ),
        remove_columns=columns,
    )
    # Rows that cannot support supervised reasoning are dropped here rather than killing the job.
    # Job 970 died 12 minutes in because a single row of ~124k had an empty `reasoning`; sparsity in
    # the corpus is not a configuration error. The count is reported so a silently-decimated arm is
    # still visible — a skip rate near 100% means the variant is wrong, not that the data is thin.
    ds = ds.filter(lambda r: not r.get("_unusable"))
    ds = ds.map(format_cafa5_for_protein_llm)
    if subset_frac < 1.0:
        ds = ds.take(max(1, int(124_000 * subset_frac)))
    return ds


def _adapt_or_mark_unusable(row, *, repo_id: str, use: str) -> dict:
    """Adapt a row, or mark it unusable so the caller can drop it and count the loss.

    Only `MissingReasoningEvidence` is caught: a row with no trace, or no evidence for its trace,
    under a reasoned target variant. Every other adapter failure is a contract violation and still
    propagates — swallowing those is how a run silently trains on the wrong thing.
    """
    from bioreason_pro.data_contract import (
        DATA_CONTRACT_ID,
        MissingReasoningEvidence,
        adapt_approved_training_row,
    )

    try:
        return {**adapt_approved_training_row(row, repo_id=repo_id, use=use), "_unusable": False}
    except MissingReasoningEvidence:
        # Same key set as a successful adapt: `datasets.map` infers one schema for the whole split,
        # so returning a short dict here would either raise or silently null out real rows.
        return {
            "prompt": {"system": "", "user": "", "assistant_reasoning": "", "assistant_answer": ""},
            "sequence": "",
            "protein_id": "",
            "go_mf": [],
            "go_bp": [],
            "go_cc": [],
            "data_contract_id": DATA_CONTRACT_ID,
            "source_dataset": repo_id,
            "source_revision": "",
            "_unusable": True,
        }


def load_rl_prompts(streaming: bool = True, seed: int = 0):
    """Stream wanglab/bioreason-pro-rl-reasoning-data (~9.2k) RL-train prompts only. Keeps the raw
    go_mf/go_bp/go_cc columns so the reward can read ground truth; adds the formatted `prompt`."""
    ds = _load_stream("wanglab/bioreason-pro-rl-reasoning-data")
    ds = ds.filter(lambda r: in_split(r["protein_id"], "rl_train", seed))
    from bioreason_pro.data_contract import adapt_approved_training_row

    columns = ds.column_names
    ds = ds.map(
        lambda r: adapt_approved_training_row(
            r, repo_id="wanglab/bioreason-pro-rl-reasoning-data", use="rl-training"
        ),
        remove_columns=columns,
    )
    return ds.map(lambda r: {**r, **{k: v for k, v in format_cafa5_for_protein_llm(r).items()
                                     if k in ("prompt", "protein_sequences", "go_aspect")}})


def load_test_set(streaming: bool = True):
    """Stream the sealed held-out test set (wanglab/bioreason-pro-test-data, ~8.63k). eval.py only."""
    from bioreason_pro.data_contract import adapt_approved_training_row

    ds = _load_stream("wanglab/bioreason-pro-test-data")
    columns = ds.column_names
    ds = ds.map(
        lambda r: adapt_approved_training_row(
            r, repo_id="wanglab/bioreason-pro-test-data", use="holdout-evaluation"
        ),
        remove_columns=columns,
    )
    return ds.map(format_cafa5_for_protein_llm)
