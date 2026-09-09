"""Fail-closed policy for model and checkpoint assets.

The repository supports only assets listed in ``approved_assets.json``. Validation functions in
this module deliberately run before importing model implementations or loading checkpoint tensors.
Unknown assets are denied: adding a model requires a reviewed manifest change and corresponding
tests.
"""

from __future__ import annotations

import json
import hashlib
import re
from importlib.resources import files
from pathlib import Path
from typing import Any


class LicensePolicyError(ValueError):
    """Raised when an asset is not approved for this repository's supported workflow."""


_IMMUTABLE_WANDB_ARTIFACT_REF = re.compile(
    r"^[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+$"
)


def require_immutable_wandb_artifact_ref(reference: str, label: str = "model artifact") -> str:
    """Require a fully-qualified, immutable W&B Artifact version.

    Aliases such as ``latest`` are intentionally rejected because they cannot reproduce a model
    hand-off. The accepted form is ``entity/project/artifact:vN``.
    """
    normalized = reference.strip()
    if not _IMMUTABLE_WANDB_ARTIFACT_REF.fullmatch(normalized):
        raise LicensePolicyError(
            f"{label} must be an immutable W&B reference in the form "
            f"'entity/project/artifact:vN'; got {reference!r}"
        )
    return normalized


def load_approved_assets() -> dict[str, Any]:
    manifest = files("bioreason_pro").joinpath("approved_assets.json")
    return json.loads(manifest.read_text(encoding="utf-8"))


def approved_model_names(kind: str) -> frozenset[str]:
    models = load_approved_assets().get("models", {})
    if kind not in models:
        raise LicensePolicyError(f"Unknown model kind {kind!r}; expected one of {sorted(models)}")
    return frozenset(models[kind])


def approved_model_spec(model_name: str, kind: str) -> dict[str, Any]:
    normalized = require_approved_model(model_name, kind)
    return load_approved_assets()["models"][kind][normalized]


def approved_model_revision(model_name: str, kind: str) -> str:
    revision = approved_model_spec(model_name, kind).get("revision")
    if not isinstance(revision, str) or not revision:
        raise LicensePolicyError(
            f"{model_name!r} has no pinned revision in approved_assets.json"
        )
    return revision


def require_approved_dataset(repo_id: str, use: str) -> str:
    """Return the pinned revision only when a dataset is approved for the requested use."""
    datasets = load_approved_assets().get("datasets", {})
    spec = datasets.get(repo_id)
    if not isinstance(spec, dict):
        raise LicensePolicyError(
            f"dataset {repo_id!r} is not recorded in approved_assets.json"
        )
    if spec.get("approved") is not True or use not in spec.get("approved_uses", []):
        pending = ", ".join(spec.get("pending_review", [])) or "dataset provenance"
        raise LicensePolicyError(
            f"dataset {repo_id!r} is not approved for {use!r}; pending review: {pending}"
        )
    revision = spec.get("revision")
    if not isinstance(revision, str) or not revision:
        raise LicensePolicyError(f"dataset {repo_id!r} has no pinned revision")
    return revision


def approved_dataset_spec(repo_id: str, use: str) -> dict[str, Any]:
    """Return an approved dataset entry after enforcing its exact intended use."""
    require_approved_dataset(repo_id, use)
    return load_approved_assets()["datasets"][repo_id]


def require_approved_local_eval_target(name: str, use: str) -> dict[str, Any]:
    """Return a locally-built (non-HF) eval target's manifest entry, only when approved for `use`.

    Mirrors `require_approved_dataset`, but for targets like `cafa_no_knowledge` (plan.md Phase 1)
    that are materialized from external APIs rather than pulled from a pinned HF dataset revision —
    so there is no single `revision` to return, only the recorded provenance for each source that
    went into the materialized file.
    """
    targets = load_approved_assets().get("local_eval_targets", {})
    spec = targets.get(name)
    if not isinstance(spec, dict):
        raise LicensePolicyError(
            f"local eval target {name!r} is not recorded in approved_assets.json"
        )
    if spec.get("approved") is not True or use not in spec.get("approved_uses", []):
        raise LicensePolicyError(
            f"local eval target {name!r} is not approved for {use!r}"
        )
    return spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_assets(
    root: str | Path = ".",
    *,
    use: str,
) -> dict[str, dict[str, Any]]:
    """Verify the exact approved GO ontology and IA weights before reward/evaluation use."""
    bundle = load_approved_assets()["reference_data"]["cafa5_evaluation_bundle"]
    if use not in bundle["approved_uses"]:
        raise LicensePolicyError(f"reference data is not approved for {use!r}")

    root_path = Path(root)
    snapshot: dict[str, dict[str, Any]] = {}
    problems = []
    for relative, spec in bundle["files"].items():
        path = root_path / relative
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != spec["size"] or digest != spec["sha256"]:
            problems.append(
                f"{path}: expected size={spec['size']} sha256={spec['sha256']}, "
                f"got size={size} sha256={digest}"
            )
            continue
        snapshot[relative] = {
            "size": size,
            "sha256": digest,
            "source": bundle["source"],
            "license": bundle["license"],
        }
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise LicensePolicyError(
            "approved GO/IA reference data is unavailable or does not match the pinned bundle:\n"
            f"{details}\nRun `python scripts/fetch_approved_reference_data.py` "
            "(add `--force` only to replace mismatched local files)."
        )
    return snapshot


def require_approved_model(model_name: str, kind: str) -> str:
    """Return a normalized model id when it is explicitly approved, otherwise fail closed."""
    normalized = model_name.strip().rstrip("/")
    allowed = approved_model_names(kind)
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise LicensePolicyError(
            f"{kind} model {model_name!r} is not approved. Supported {kind} models: {choices}. "
            "Add a reviewed entry to bioreason_pro/approved_assets.json before using a new asset."
        )
    return normalized


def require_no_unapproved_go_assets(go_embeddings_path: str = "") -> None:
    """Block precomputed GO embeddings until their source-model provenance is recorded."""
    if go_embeddings_path:
        raise LicensePolicyError(
            "Precomputed GO embeddings are not yet approved. Record the source model, revision, "
            "license, generation procedure, and checksum in approved_assets.json first."
        )


def _read_required_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LicensePolicyError(f"{path.parent}: missing required {label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LicensePolicyError(f"{path}: unreadable {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LicensePolicyError(f"{path}: {label} must contain a JSON object")
    return value


def validate_checkpoint_layout(
    checkpoint_dir: str | Path,
    *,
    expected_text_model: str | None = None,
    expected_protein_model: str | None = None,
    _chain_depth: int = 0,
) -> dict[str, Any]:
    """Validate a reloadable adapter checkpoint without loading any model or tensor data.

    A supported multimodal checkpoint is an adapter over an approved text model plus a run snapshot
    naming the approved protein model and the trained projection weights. Full-model layouts and
    legacy checkpoints with missing metadata are rejected instead of being guessed from defaults.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise LicensePolicyError(f"{root}: checkpoint directory does not exist")
    if (root / "protein_model").exists():
        raise LicensePolicyError(
            f"{root}: bundled protein_model checkpoints are unsupported; only ESM2 adapter "
            "checkpoints with external approved base models may be evaluated"
        )

    adapter = _read_required_json(root / "adapter_config.json", "adapter_config.json")
    run_args = _read_required_json(root / "run_args.json", "run_args.json")
    data_manifest = _read_required_json(root / "data_manifest.json", "data_manifest.json")
    if not (root / "projections.pt").is_file():
        raise LicensePolicyError(f"{root}: missing required projections.pt")

    approved_contract = load_approved_assets()["data_contract"]
    if data_manifest.get("contract_id") != approved_contract["id"]:
        raise LicensePolicyError(
            f"{root / 'data_manifest.json'}: unsupported or missing data contract"
        )
    if data_manifest.get("schema_version") != approved_contract["schema_version"]:
        raise LicensePolicyError(
            f"{root / 'data_manifest.json'}: incompatible data-contract schema"
        )
    if data_manifest.get("approved_prompt_fields") != approved_contract["approved_prompt_fields"]:
        raise LicensePolicyError(
            f"{root / 'data_manifest.json'}: prompt fields do not match the approved contract"
        )
    if data_manifest.get("approved_label_fields") != approved_contract["approved_label_fields"]:
        raise LicensePolicyError(
            f"{root / 'data_manifest.json'}: label fields do not match the approved contract"
        )
    expected_preprocessing = {
        "max_protein_residues": 1024,
        "protein_special_tokens": ["BOS", "EOS"],
        "max_protein_tokens": 1026,
    }
    if data_manifest.get("preprocessing") != expected_preprocessing:
        raise LicensePolicyError(
            f"{root / 'data_manifest.json'}: protein preprocessing does not match ESM2 limits"
        )

    base_model = adapter.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise LicensePolicyError(
            f"{root / 'adapter_config.json'}: base_model_name_or_path is required"
        )
    base_model = require_approved_model(base_model, "text")

    text_model = run_args.get("model_name") or "Qwen/Qwen3-4B-Thinking-2507"
    text_model = require_approved_model(str(text_model), "text")
    text_revision = approved_model_revision(text_model, "text")
    if run_args.get("text_model_revision") != text_revision:
        raise LicensePolicyError(
            f"{root / 'run_args.json'}: text_model_revision must be {text_revision!r}"
        )
    if base_model != text_model:
        raise LicensePolicyError(
            f"{root}: adapter base model {base_model!r} does not match run_args model {text_model!r}"
        )
    if expected_text_model is not None:
        expected_text_model = require_approved_model(expected_text_model, "text")
        if text_model != expected_text_model:
            raise LicensePolicyError(
                f"{root}: checkpoint text model {text_model!r} does not match requested "
                f"{expected_text_model!r}"
            )
    if run_args.get("use_multimodal") is not True:
        raise LicensePolicyError(f"{root / 'run_args.json'}: use_multimodal must be true")
    protein_model = run_args.get("esm_model_name")
    if not isinstance(protein_model, str) or not protein_model.strip():
        raise LicensePolicyError(f"{root / 'run_args.json'}: esm_model_name is required")
    protein_model = require_approved_model(protein_model, "protein")
    protein_revision = approved_model_revision(protein_model, "protein")
    if run_args.get("esm_model_revision") != protein_revision:
        raise LicensePolicyError(
            f"{root / 'run_args.json'}: esm_model_revision must be {protein_revision!r}"
        )
    if expected_protein_model is not None:
        expected_protein_model = require_approved_model(expected_protein_model, "protein")
        if protein_model != expected_protein_model:
            raise LicensePolicyError(
                f"{root}: checkpoint protein model {protein_model!r} does not match requested "
                f"{expected_protein_model!r}"
            )

    if run_args.get("go_cached_embedding_path"):
        raise LicensePolicyError(
            f"{root / 'run_args.json'}: legacy go_cached_embedding_path is not approved"
        )
    if run_args.get("asset_manifest_schema_version") != load_approved_assets()["schema_version"]:
        raise LicensePolicyError(
            f"{root / 'run_args.json'}: asset_manifest_schema_version is missing or incompatible"
        )
    stage = run_args.get("stage")
    if stage not in {"sft", "rl"}:
        raise LicensePolicyError(f"{root / 'run_args.json'}: stage must be 'sft' or 'rl'")
    if stage == "rl":
        if _chain_depth:
            raise LicensePolicyError(f"{root}: nested RL checkpoint chains are unsupported")
        require_immutable_wandb_artifact_ref(
            str(run_args.get("sft_artifact") or ""),
            "RL checkpoint sft_artifact",
        )
        if run_args.get("resolved_sft_artifact") != run_args.get("sft_artifact"):
            raise LicensePolicyError(
                f"{root / 'run_args.json'}: resolved_sft_artifact must match sft_artifact"
            )
        if run_args.get("sft_adapter") != "sft_adapter":
            raise LicensePolicyError(
                f"{root / 'run_args.json'}: RL checkpoints must use bundled sft_adapter"
            )
        sft_args = validate_checkpoint_layout(
            root / "sft_adapter",
            expected_text_model=text_model,
            expected_protein_model=protein_model,
            _chain_depth=_chain_depth + 1,
        )
        if sft_args.get("stage") != "sft":
            raise LicensePolicyError(f"{root / 'sft_adapter'}: expected an SFT checkpoint")
    return run_args
