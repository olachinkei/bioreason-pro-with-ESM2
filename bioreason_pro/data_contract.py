"""Approved, minimal data contract for SFT, RL, and public holdout evaluation.

Only protein identifiers, amino-acid sequences, and GO label columns are used by default. Additional
context columns (GPT reasoning, GO-GPT, PPI, InterPro, and free-text metadata) are excluded from the
supported training path unless explicitly adopted below (each adoption is an owner-approved gate R3
decision, recorded in `bioreason_pro/approved_assets.json`'s `column_provenance`).
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from bioreason_pro.license_policy import load_approved_assets, require_approved_dataset

DATA_CONTRACT_ID = "sequence-go-labels-v1"
DATA_CONTRACT_SCHEMA_VERSION = 2
PUBLIC_HOLDOUT_IDS_PATH = Path(__file__).resolve().parents[1] / "data" / "public_holdout_ids.txt"
_GO_RE = re.compile(r"GO:\d{7}")

APPROVED_SOURCE_FIELDS = frozenset(
    {"protein_id", "sequence", "go_mf", "go_bp", "go_cc"}
)
APPROVED_PROMPT_FIELDS = ("sequence",)
APPROVED_LABEL_FIELDS = ("go_mf", "go_bp", "go_cc")


def parse_go_terms(value: Any) -> set[str]:
    """Parse GO ids from source list columns or stringified public-holdout columns."""
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, str) and _GO_RE.fullmatch(item)}
    if isinstance(value, str):
        return set(_GO_RE.findall(value))
    return set()


def go_labels(row: dict[str, Any]) -> dict[str, list[str]]:
    """Return deterministic, aspect-separated GO labels from approved columns only."""
    return {
        aspect: sorted(parse_go_terms(row.get(column)))
        for aspect, column in (("MF", "go_mf"), ("BP", "go_bp"), ("CC", "go_cc"))
    }


# Supervision-target variants, selected by SENPAI_TARGET_VARIANT.
#
# "full_closure" (default) writes the label columns as they ship. Those columns are ancestor-closed
# by the True-Path Rule, so on the RL split a target carries a mean of 22.9 GO ids of which only 3.2
# are leaves — 86.1% of the supervision is terms the scorer re-derives on its own (cafaeval
# propagates predictions). The model learns that shape faithfully: Phase 4 rollouts emit a mean of
# 23.9 ids, and removing the implied ancestors from a prediction moves weighted F_max by 0.000000.
#
# "leaf_only" writes just the most specific terms. Propagation reconstructs the same closure at
# scoring time, so a correct leaf still earns its ancestors, but the fixed generation budget now
# carries ~3 decisions instead of ~23 mostly-redundant ones.
# "leaf_mf_only" follows from job 627, where leaf_only lifted MF by 99.5% while BP fell 22.9% and CC
# fell 42.4%. Scoring propagates both sides, so a correct leaf earns its whole ancestor chain: that
# is free for shallow MF annotations, but in the deep BP and CC hierarchies leaf-only supervision
# trains the model never to emit the mid-level term it could actually be confident about, and a
# missed leaf then earns nothing where a hedge would have earned partial credit.
# "leaf_only_reasoned" (ADR-017) additionally supervises the <think> block from the dataset's
# `reasoning` column, and — inseparably — puts the evidence that reasoning cites into the prompt.
# The coupling is the safety property, not a convenience: sampled 20/20, those traces cite InterPro
# domain ids with residue ranges and STRING interaction partners. Supervising them against the
# sequence-only prompt would train the model to INVENT `IPR051630 (residues 9-1410)` from an amino
# acid string — confident, specific, and unknowable. A reviewer in drug discovery would have no way
# to tell that apart from a real finding, which is the precise failure ADR-014 exists to avoid. So
# the variant cannot be configured into the dangerous half.
TARGET_VARIANTS = ("full_closure", "leaf_only", "leaf_mf_only", "leaf_only_reasoned")
DEFAULT_TARGET_VARIANT = "full_closure"

# Which aspects get leaf-only treatment, per variant. Absent aspects keep the shipped closure.
LEAF_ASPECTS: dict[str, frozenset[str]] = {
    "full_closure": frozenset(),
    "leaf_only": frozenset({"MF", "BP", "CC"}),
    "leaf_mf_only": frozenset({"MF"}),
    "leaf_only_reasoned": frozenset({"MF", "BP", "CC"}),
}

# Variants that supervise the reasoning trace, and therefore require the evidence columns in the
# prompt. Kept as data so the coupling is checkable by a test rather than remembered.
REASONED_TARGET_VARIANTS = ("leaf_only_reasoned",)


class MissingReasoningEvidence(ValueError):
    """An SFT row cannot support supervised reasoning: no trace, or no evidence for it.

    A distinct type so the SFT loader can SKIP the row and count it, rather than either killing a
    3.5-hour job over data sparsity or - much worse - silently supervising an empty <think>.
    """


# Columns admitted to the prompt only for a reasoned variant, in the order they are rendered.
# Every one of these is cited by the traces; without them the supervision is unlearnable.
#
# `go_pred` (ADR-037): GO-GPT's own greedy predictions, full ancestor closure, per-aspect text. Paper
# parity — upstream's own BioReason-Pro takes these as a prompt input (refining a strong prior, not
# generating from scratch). Deferred from Phase 5 (plan.md, gate R3, 2026-08-23 "baseline only for
# now") pending the GO-GPT baseline measurement (ADR-033: 0.50059) and a license check (this session:
# `wanglab/gogpt` weights Apache-2.0, not gated; vendored code MIT — no licence blocker). A 60-trace
# spot check found 56/60 reasoning traces cite a GO id that also appears in `go_pred` verbatim — unlike
# organism (never cited, see ORGANISM_COLUMN below), this is real evidence traces build on, so it is
# gated like the rest of CONTEXT_COLUMNS rather than rendered like organism.
CONTEXT_COLUMNS = (
    ("interpro_formatted", "Domain annotations (InterPro)"),
    ("ppi_formatted", "Interaction partners (STRING)"),
    ("subcellular_location", "Subcellular location (UniProtKB)"),
    ("go_pred", "Predicted GO terms (GO-GPT)"),
)

# plan.md Phase 5: the paper's prompt carries organism, ours didn't. Rendered the same way as
# CONTEXT_COLUMNS (a stable "not available" placeholder, reasoned-variant only), but kept out of that
# tuple deliberately: CONTEXT_COLUMNS is specifically evidence sampled traces are known to CITE, and a
# missing entry there gates SFT supervision closed (MissingReasoningEvidence). A spot check of 60 real
# reasoning traces found zero that name their own protein's organism or genus -- this is a supplementary
# input signal for the GO-term prediction itself, not evidence a trace could be caught fabricating.
ORGANISM_COLUMN = ("organism", "Organism (UniProtKB)")


def active_target_variant() -> str:
    """Read SENPAI_TARGET_VARIANT, failing closed on an unknown name."""
    import os

    name = os.environ.get("SENPAI_TARGET_VARIANT", DEFAULT_TARGET_VARIANT).strip()
    if name not in TARGET_VARIANTS:
        raise ValueError(
            f"SENPAI_TARGET_VARIANT={name!r} is not a known target variant; "
            f"expected one of {list(TARGET_VARIANTS)}"
        )
    return name


def require_active_variant_matches(expected: str | None, *, checkpoint_label: str = "checkpoint") -> None:
    """Fail closed if the active SENPAI_TARGET_VARIANT does not match what a checkpoint was trained
    with (its producing run's own `target_variant` config value, e.g. from `wandb.Api()`).

    An unset SENPAI_TARGET_VARIANT silently defaults to DEFAULT_TARGET_VARIANT ("full_closure"),
    which renders a prompt with NO InterPro/PPI/subcellular-location context at all -- a checkpoint
    trained on `leaf_only_reasoned` evaluated this way still reasons (that behaviour is trained in)
    but with no evidence to reason FROM, so it fabricates domain claims instead of leaving them out.
    This produced a plausible-looking but meaningless eval run once already (plan.md ADR-031's first,
    retracted Phase 3 sweep) before anyone noticed the traces cited no real evidence. `expected=None`
    (the producing run never recorded a target_variant, e.g. older runs predating this being logged)
    skips the check rather than blocking on missing historical metadata.
    """
    if expected is None:
        return
    active = active_target_variant()
    if active != expected:
        raise ValueError(
            f"{checkpoint_label}: SENPAI_TARGET_VARIANT={active!r} does not match this checkpoint's "
            f"recorded target_variant={expected!r}. Set SENPAI_TARGET_VARIANT={expected!r} before "
            "evaluating it, or the prompt shape will not match what it was trained on."
        )


@lru_cache(maxsize=1)
def _go_ancestors_for_targets() -> dict[str, set[str]]:
    """Ancestors for leaf-only authoring. Cached: targets are formatted per row while streaming."""
    from bioreason_pro.go_obo import load_go_ancestors

    return load_go_ancestors(str(Path(__file__).resolve().parents[1] / "data" / "go-basic.obo"))


def format_go_answer(row: dict[str, Any], variant: str | None = None) -> str:
    """Create an authored target from approved GO ids instead of source-generated prose."""
    name = variant if variant is not None else active_target_variant()
    if name not in TARGET_VARIANTS:
        raise ValueError(f"unknown target variant {name!r}; expected one of {list(TARGET_VARIANTS)}")
    labels = go_labels(row)
    leaf_aspects = LEAF_ASPECTS[name]
    if leaf_aspects:
        from bioreason_pro.rewards import strip_implied_ancestors

        ancestors = _go_ancestors_for_targets()
        # GO ancestry does not cross MF/BP/CC, so per-aspect stripping equals stripping the union.
        labels = {
            aspect: (
                sorted(strip_implied_ancestors(set(terms), ancestors))
                if aspect in leaf_aspects
                else terms
            )
            for aspect, terms in labels.items()
        }
    lines = [f"{aspect}: {', '.join(terms)}" for aspect, terms in labels.items() if terms]
    return "\n".join(lines) if lines else "No approved GO annotations."


def _has_known_protein_function(protein_function: str | None) -> bool:
    text = (protein_function or "").strip()
    return bool(text) and "not known" not in text.lower()


def functional_summary_instruction(protein_function: str | None) -> str:
    """The extra instruction sentence asking for a functional summary, gated on whether one is
    actually knowable for this row.

    `protein_function` reads "Not known" for a large share of rows (plan.md Phase 4 corpus audit).
    Asking the model to summarize a function it is then shown as "not known" trains a contradiction
    -- mirrors upstream's own `force_uniprot_summary` rationale (bowang-lab/BioReason-Pro,
    dataset/cafa5/load.py): ask for a summary only when the target actually has one to give.
    """
    if _has_known_protein_function(protein_function):
        return " Also give a brief functional summary of the protein, in UniProt style."
    return " The protein's function is not known; do not attempt a functional summary."


def compose_functional_summary(final_answer: str, protein_function: str | None) -> str:
    """Splice UniProt's raw Function annotation into the authored answer as its own line.

    Mirrors upstream's `_add_uniprot_summary` (bowang-lab/BioReason-Pro, dataset/cafa5/load.py):
    keep `final_answer`'s first line as a header, insert the UniProt text as the next line, then the
    remaining lines unchanged.
    """
    lines = (final_answer or "").split("\n")
    summary = (protein_function or "").strip()
    return "\n".join([lines[0], f"- UniProt Summary: {summary}", *lines[1:]])


def adapt_approved_training_row(
    row: dict[str, Any],
    *,
    repo_id: str,
    use: str,
    max_length_protein: int = 1024,
) -> dict[str, Any]:
    """Convert a source row to the repository prompt shape using only approved fields."""
    revision = require_approved_dataset(repo_id, use)
    return _adapt_fields(
        row,
        source_dataset=repo_id,
        source_revision=revision,
        max_length_protein=max_length_protein,
        use=use,
    )


def adapt_row_for_evaluation(
    row: dict[str, Any],
    *,
    source_dataset: str = "external-eval-target",
    max_length_protein: int = 1024,
) -> dict[str, Any]:
    """Same prompt/label shape as `adapt_approved_training_row`, for eval targets whose approval is
    checked elsewhere (e.g. a `source="local"` eval_targets target, gated by
    `require_approved_local_eval_target`) rather than through `require_approved_dataset` on a pinned
    HF revision — so there is no `repo_id` to check here.

    `use="evaluation"` (not `sft*`) means the reasoning-supervision requirement in `_adapt_fields`
    never fires: eval rows carry no assistant turn to supervise, only a prompt to score against. The
    reasoned-variant context rendering (CONTEXT_COLUMNS, "not available" placeholders) still applies
    exactly as it does for `bioreason_pro_test` via `adapt_approved_training_row` — this exists so a
    second target can reproduce that same prompt shape without duplicating this function's logic.
    """
    return _adapt_fields(
        row,
        source_dataset=source_dataset,
        source_revision="n/a",
        max_length_protein=max_length_protein,
        use="evaluation",
    )


def adapt_synthetic_fixture_row(
    row: dict[str, Any], *, max_length_protein: int = 1024, use: str = "sft-training"
) -> dict[str, Any]:
    """Apply the same field contract to a checked-in, intentionally synthetic row."""
    return _adapt_fields(
        row,
        source_dataset="tests/fixtures/approved_training_rows.jsonl",
        source_revision="checked-in",
        max_length_protein=max_length_protein,
        use=use,
    )


def _adapt_fields(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_revision: str,
    max_length_protein: int,
    use: str = "sft-training",
) -> dict[str, Any]:
    protein_id = str(row.get("protein_id") or "").strip()
    sequence = str(row.get("sequence") or "").strip()[:max_length_protein]
    if not protein_id or not sequence:
        raise ValueError(f"{source_dataset}: row requires non-empty protein_id and sequence")

    labels = go_labels(row)
    answer = format_go_answer(row)
    variant = active_target_variant()
    reasoned = variant in REASONED_TARGET_VARIANTS
    instruction = (
        "Predict the Gene Ontology terms for this protein. Return only the approved MF, BP, and CC "
        "GO identifiers grouped by aspect."
    )
    if reasoned:
        instruction = instruction + functional_summary_instruction(row.get("protein_function"))
    user = instruction + f"\n\nProtein sequence:\n{sequence}"
    reasoning = ""
    if reasoned:
        # EVERY context section is rendered at every stage, with an explicit "not available" when a
        # source is missing. The sealed test set has no `ppi_formatted` while 20/20 training traces
        # cite STRING partners, so silently dropping the section would train one prompt shape and
        # evaluate another — and a model trained to always see partners is exactly the one that will
        # invent them when they vanish. A stable placeholder teaches "absent" as a real state.
        organism_column, organism_label = ORGANISM_COLUMN
        organism_value = str(row.get(organism_column) or "").strip()
        blocks = [f"{organism_label}: {organism_value}" if organism_value
                  else f"{organism_label}: not available"]
        for column, label in CONTEXT_COLUMNS:
            value = str(row.get(column) or "").strip()
            blocks.append(f"{label}:\n{value}" if value else f"{label}:\nnot available")
        user = user + "\n\n" + "\n\n".join(blocks)

        # Paper parity (plan.md Phase 4): splice a UniProt functional summary ahead of the GO-term
        # lines, exactly where upstream's own `_add_uniprot_summary` puts it. `final_answer` is
        # model-generated prose (owner-approved for adoption without scoring, R3); `protein_function`
        # is UniProt's own curated Function text, spliced in verbatim the way upstream splices it.
        final_answer_text = str(row.get("final_answer") or "").strip()
        if final_answer_text:
            summary = compose_functional_summary(final_answer_text, row.get("protein_function"))
            answer = summary + "\n\n" + answer

        # The trace is only SUPERVISED during SFT, so that is the only stage where its evidence must
        # actually be present. RL scores its own rollouts and evaluation has no target at all; the
        # rl-reasoning dataset's schema has no `reasoning` column at all (not merely empty cells —
        # confirmed absent from the column list itself), so requiring one there would make the
        # reasoned variant unusable for RL. Using this dataset for RL training is an explicit owner
        # decision (plan.md ADR-026, 2026-08-21) taken with that gap known, not an oversight.
        if use.startswith("sft"):
            if not any(str(row.get(c) or "").strip() for c, _ in CONTEXT_COLUMNS):
                raise MissingReasoningEvidence(
                    f"{source_dataset}: target variant {variant!r} supervises reasoning, but this "
                    f"row carries none of {[c for c, _ in CONTEXT_COLUMNS]}. Training the trace "
                    "without the evidence it cites would teach the model to fabricate domain and "
                    "interaction facts."
                )
            reasoning = str(row.get("reasoning") or "").strip()
            if not reasoning:
                raise MissingReasoningEvidence(
                    f"{source_dataset}: target variant {variant!r} requires a non-empty `reasoning` "
                    "column; an empty one would reintroduce the manufactured empty <think> "
                    "(ADR-015)."
                )
            if not final_answer_text:
                raise MissingReasoningEvidence(
                    f"{source_dataset}: target variant {variant!r} now supervises a functional "
                    "summary spliced from `final_answer` (plan.md Phase 4), but this row has none. "
                    "Training the summary section without a target would teach the model to "
                    "fabricate one."
                )
    return {
        "prompt": {
            "system": "You are a protein function prediction assistant.",
            "user": user,
            "assistant_reasoning": reasoning,
            "assistant_answer": answer,
        },
        "sequence": sequence,
        "protein_id": protein_id,
        "go_mf": labels["MF"],
        "go_bp": labels["BP"],
        "go_cc": labels["CC"],
        "data_contract_id": DATA_CONTRACT_ID,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
    }


@lru_cache(maxsize=1)
def public_holdout_ids() -> frozenset[str]:
    """Load the pinned public-holdout ID set committed for leakage prevention."""
    if not PUBLIC_HOLDOUT_IDS_PATH.is_file():
        raise FileNotFoundError(
            f"{PUBLIC_HOLDOUT_IDS_PATH} is required; regenerate it with "
            "scripts/audit_data_contract.py --write"
        )
    ids = frozenset(
        line.strip()
        for line in PUBLIC_HOLDOUT_IDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest = load_approved_assets()["data_contract"]["public_holdout_ids"]
    digest = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    if len(ids) != manifest["count"] or digest != manifest["sha256"]:
        raise ValueError(
            "public_holdout_ids.txt does not match approved_assets.json "
            f"(count={len(ids)}, sha256={digest})"
        )
    return ids


def is_public_holdout(protein_id: str) -> bool:
    return protein_id in public_holdout_ids()


def assert_train_val_holdout_disjoint(
    train_ids: Iterable[str], validation_ids: Iterable[str], holdout_ids: Iterable[str]
) -> None:
    """Fail if the three operational split sets overlap."""
    train, validation, holdout = set(train_ids), set(validation_ids), set(holdout_ids)
    overlaps = {
        "train∩validation": train & validation,
        "train∩public_holdout": train & holdout,
        "validation∩public_holdout": validation & holdout,
    }
    bad = {name: sorted(values)[:10] for name, values in overlaps.items() if values}
    if bad:
        raise AssertionError(f"data-contract leakage detected: {bad}")


def build_run_data_manifest(stage: str, smoke: bool, seed: int) -> dict[str, Any]:
    """Snapshot every row source and prompt/label field used by a run."""
    manifest = load_approved_assets()
    if smoke:
        source = {
            "kind": "checked-in-synthetic-fixture",
            "path": "tests/fixtures/approved_training_rows.jsonl",
        }
    else:
        repo_id = {
            "sft": "wanglab/bioreason-pro-sft-reasoning-data",
            "rl": "wanglab/bioreason-pro-rl-reasoning-data",
        }[stage]
        spec = manifest["datasets"][repo_id]
        source = {
            "kind": "huggingface",
            "repo_id": repo_id,
            "revision": spec["revision"],
            "hf_split": spec["hf_split"],
            "approved_use": f"{stage}-training",
            "parquet_files": spec["parquet_files"],
        }
    return {
        "schema_version": DATA_CONTRACT_SCHEMA_VERSION,
        "contract_id": DATA_CONTRACT_ID,
        "source": source,
        "approved_prompt_fields": list(APPROVED_PROMPT_FIELDS),
        "approved_label_fields": list(APPROVED_LABEL_FIELDS),
        "excluded_source_fields": manifest["data_contract"]["excluded_source_fields"],
        "preprocessing": {
            "max_protein_residues": 1024,
            "protein_special_tokens": ["BOS", "EOS"],
            "max_protein_tokens": 1026,
        },
        "split_policy": {
            "seed": seed,
            "public_holdout_excluded_first": True,
            "hash_function": "sha256",
            "internal_splits": {"rl_train": 0.8, "val": 0.1, "reserved_test": 0.1},
            "public_holdout_ids": manifest["data_contract"]["public_holdout_ids"],
        },
    }


def data_contract_json(stage: str, smoke: bool, seed: int) -> str:
    return json.dumps(build_run_data_manifest(stage, smoke, seed), sort_keys=True)
