#!/usr/bin/env python3
"""Inventory artifacts derived from ESM3, ESM-C 600M, or the released ESM3 paper checkpoint.

Phase 0 of `plan.md` requires an inventory of incompatible assets before they can be quarantined
or deleted. The code-side boundary is already enforced (see `bioreason_pro/license_policy.py` and
`tests/test_license_policy.py`); what this script finds is the *residue* of the pre-boundary era:
git history, W&B artifacts, and files on local or cluster storage.

Scanners are independent and each may be skipped, so the same script runs on a laptop (git plus
local paths) and on the cluster (shared storage) and produces the same report shape.

Findings are classified, not silently dropped. A file that names ESM3 in order to *forbid* it is
recorded as `policy_reference`; anything that names it as an asset to load is `asset_reference`.
Only the latter needs quarantine.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "data" / "incompatible_asset_inventory.json"

# Reviewed denial patterns. `program.md` states the boundary in prose; this is its machine form.
INCOMPATIBLE_PATTERNS: dict[str, re.Pattern[str]] = {
    # EvolutionaryScale ESM3 weights in any packaging.
    "esm3": re.compile(r"esm3[\w.-]*", re.IGNORECASE),
    # ESM-C 600M weights. Matches esmc_600m / esm-c / esm_c but not the approved facebook/esm2_*.
    "esmc": re.compile(r"esm[-_]?c(?:[-_]?600m|\b)", re.IGNORECASE),
    # The released BioReason-Pro checkpoint (contains ESM3). The *-data repos are approved and must
    # not match, hence the explicit boundary after the stage name.
    "released_paper_checkpoint": re.compile(
        r"wanglab/bioreason-pro-(?:rl|sft)(?![\w-])", re.IGNORECASE
    ),
    # Precomputed GO embeddings whose source model is unrecorded (license_policy denies these).
    "unapproved_go_embeddings": re.compile(r"go_cached_embedding_path\s*[:=]\s*[\"']?[^\s\"',}]+"),
}

# Filesystem-only rules. These would produce noise against the repository itself: `cafa5` appears
# legitimately in `approved_assets.json` (the approved GO/IA evaluation bundle) and in the
# intentionally-disabled `eval_targets/cafa5.py`. On disk, a cached copy of the gated dataset is a
# different matter — `plan.md` keeps CAFA5 unavailable until its gate and provenance review pass.
FILESYSTEM_ONLY_PATTERNS: dict[str, re.Pattern[str]] = {
    "gated_dataset_cache": re.compile(r"(?:datasets--wanglab--cafa5|wanglab/cafa5)", re.IGNORECASE),
}

# Paths whose whole purpose is to name the forbidden assets in order to deny them.
POLICY_PATHS = (
    "bioreason_pro/license_policy.py",
    "bioreason_pro/approved_assets.json",
    # This scanner's own committed output. It names every incompatible asset by construction; left
    # unlisted, each run would flag the previous run's report and never converge.
    "data/incompatible_asset_inventory.json",
    "agent_policy.yaml",
    "slurm/agent_policy.py",
    "program.md",
    "plan.md",
    "README.md",
    "RUNBOOK.md",
    "scripts/inventory_incompatible_assets.py",
)
POLICY_PREFIXES = ("tests/", "instructions/", "agent_iterations/")

GIT_SCAN_GLOBS = ("*.py", "*.yaml", "*.yml", "*.json", "*.sh", "*.sbatch", "*.tmpl", "*.md")


# Strongest classification wins when a file mixes kinds of match.
CLASSIFICATION_RANK = {"policy_reference": 0, "commentary": 1, "asset_reference": 2}
COMMENT_PREFIXES = ("#", "//", "--", "/*", "*")


def _classify(path: str, text: str) -> str:
    """Classify one matching line.

    `policy_reference` — the file exists to deny these assets (manifest, policy code, tests, docs).
    `commentary` — a comment or prose line that merely names them, e.g. "not comparable to ESM3".
    `asset_reference` — anything else, i.e. a line that could actually route to the asset.
    """
    if path in POLICY_PATHS or path.startswith(POLICY_PREFIXES):
        return "policy_reference"
    stripped = text.strip()
    if stripped.startswith(COMMENT_PREFIXES) or path.endswith(".md"):
        return "commentary"
    return "asset_reference"


def _combined_pattern() -> str:
    return "|".join(pattern.pattern for pattern in INCOMPATIBLE_PATTERNS.values())


def _matched_rules(text: str, *, include_filesystem_only: bool = False) -> list[str]:
    rules = dict(INCOMPATIBLE_PATTERNS)
    if include_filesystem_only:
        rules.update(FILESYSTEM_ONLY_PATTERNS)
    return sorted(name for name, pattern in rules.items() if pattern.search(text))


def _git(*args: str, allow_empty_match: bool = False) -> str:
    """Run git and fail loudly.

    A scanner that silently returns nothing is worse than one that crashes: it reads as `clean`.
    `git grep` exits 1 when it matches nothing, which is the one non-zero status we accept.
    """
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    if result.returncode and not (allow_empty_match and result.returncode == 1):
        raise SystemExit(
            f"git {' '.join(args)} failed with status {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def _refs() -> list[str]:
    output = _git(
        "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes/origin"
    )
    return [ref for ref in output.splitlines() if ref and not ref.endswith("/HEAD")]


def scan_git() -> list[dict[str, Any]]:
    """Scan every branch for references to incompatible assets.

    Findings are aggregated per (ref, path): a single file naming ESM3 on forty lines is one thing
    to quarantine, not forty. Line numbers stay in `first_lines` so a reviewer can jump in.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    pattern = _combined_pattern()
    for ref in _refs():
        # -P (PCRE): the denial patterns use lookahead, which POSIX ERE cannot express.
        output = _git(
            "grep", "--no-color", "-I", "-n", "-P", "-i", pattern, ref, "--", *GIT_SCAN_GLOBS,
            allow_empty_match=True,
        )
        for line in output.splitlines():
            # `ref:path:lineno:content`
            ref_name, _, content = line.partition(":")
            path, _, rest = content.partition(":")
            lineno, _, text = rest.partition(":")
            if not lineno.isdigit():
                continue
            key = (ref_name, path)
            entry = grouped.setdefault(
                key,
                {
                    "source": "git",
                    "ref": ref_name,
                    "path": path,
                    "matches": 0,
                    "rules": set(),
                    "first_lines": [],
                    "classification": "policy_reference",
                    "evidence": text.strip()[:200],
                },
            )
            entry["matches"] += 1
            entry["rules"].update(_matched_rules(text))
            if len(entry["first_lines"]) < 5:
                entry["first_lines"].append(int(lineno))
            classification = _classify(path, text)
            if CLASSIFICATION_RANK[classification] > CLASSIFICATION_RANK[entry["classification"]]:
                entry["classification"] = classification
                entry["evidence"] = text.strip()[:200]
    findings = []
    for entry in grouped.values():
        entry["rules"] = sorted(entry["rules"])
        findings.append(entry)
    return findings


def summarize_git_branches(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-branch rollup: the unit a reviewer actually acts on."""
    merged = {
        line.strip().lstrip("* ").strip()
        for line in _git("branch", "-a", "--merged", "origin/main").splitlines()
        if line.strip()
    }

    def _is_merged(ref: str) -> bool:
        short = ref.replace("refs/heads/", "").replace("refs/remotes/", "")
        return short in merged or f"remotes/{short}" in merged

    rollup: dict[str, dict[str, Any]] = {}
    for item in findings:
        if item["source"] != "git" or item["classification"] != "asset_reference":
            continue
        entry = rollup.setdefault(
            item["ref"],
            {
                "ref": item["ref"],
                "files": 0,
                "matches": 0,
                "rules": set(),
                "merged_into_main": _is_merged(item["ref"]),
            },
        )
        entry["files"] += 1
        entry["matches"] += item["matches"]
        entry["rules"].update(item["rules"])
    for entry in rollup.values():
        entry["rules"] = sorted(entry["rules"])
    return sorted(rollup.values(), key=lambda entry: (-entry["matches"], entry["ref"]))


def _local_roots(extra: Iterable[str]) -> list[Path]:
    roots = [
        ROOT / "wandb",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "torch" / "hub",
    ]
    roots.extend(Path(item).expanduser() for item in extra)
    return [root for root in roots if root.exists()]


def scan_local(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Scan local and mounted-cluster storage for incompatible model caches and run metadata."""
    findings: list[dict[str, Any]] = []
    for root in roots:
        # A cache directory that matches by name is one thing to quarantine; listing every blob
        # underneath it adds no information, so children of a reported match are collapsed.
        reported: list[str] = []
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            rules = _matched_rules(relative, include_filesystem_only=True)
            if rules and any(relative.startswith(prefix + "/") for prefix in reported):
                continue
            if rules:
                reported.append(relative)
                findings.append(
                    {
                        "source": "filesystem",
                        "root": str(root),
                        "path": relative,
                        "rules": rules,
                        "classification": "asset_reference",
                        "evidence": "path name",
                    }
                )
            # W&B run metadata records the models a run actually loaded.
            if path.name in {"config.yaml", "wandb-metadata.json"} and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rules = _matched_rules(text, include_filesystem_only=True)
                if rules:
                    findings.append(
                        {
                            "source": "filesystem",
                            "root": str(root),
                            "path": relative,
                            "rules": rules,
                            "classification": "asset_reference",
                            "evidence": "run metadata names an incompatible asset",
                        }
                    )
    return findings


def scan_wandb(project: str) -> list[dict[str, Any]]:
    """Scan a W&B project's artifacts and their producing runs. Requires read access."""
    import wandb

    api = wandb.Api()
    findings: list[dict[str, Any]] = []
    for collection_type in api.artifact_types(project):
        for collection in collection_type.collections():
            for artifact in collection.artifacts():
                blob = json.dumps(
                    {
                        "name": artifact.name,
                        "type": artifact.type,
                        "metadata": artifact.metadata,
                        "aliases": list(artifact.aliases),
                    },
                    default=str,
                )
                rules = _matched_rules(blob)
                if rules:
                    findings.append(
                        {
                            "source": "wandb",
                            "project": project,
                            "artifact": f"{artifact.name}",
                            "artifact_type": artifact.type,
                            "rules": rules,
                            "classification": "asset_reference",
                            "evidence": "artifact metadata names an incompatible asset",
                        }
                    )
    for run in api.runs(project):
        blob = json.dumps({"config": run.config, "name": run.name}, default=str)
        rules = _matched_rules(blob)
        if not rules:
            continue
        findings.append(
            {
                "source": "wandb",
                "project": project,
                "run": run.id,
                "run_name": run.name,
                "rules": rules,
                "classification": "asset_reference",
                "evidence": "run config names an incompatible asset",
            }
        )
        # The run config is the lead; its outputs are what actually need quarantine. An artifact
        # whose producing run loaded ESM3/ESM-C is derived from it regardless of its own metadata.
        for artifact in run.logged_artifacts():
            # `wandb-history`/`wandb-*` are W&B's own telemetry, not derived model assets.
            if artifact.type.startswith("wandb-"):
                continue
            findings.append(
                {
                    "source": "wandb",
                    "project": project,
                    "artifact": artifact.name,
                    "artifact_type": artifact.type,
                    "derived_from_run": run.id,
                    "rules": rules,
                    "classification": "asset_reference",
                    "evidence": f"logged by run {run.id}, whose config names an incompatible asset",
                }
            )
    return findings


def build_report(
    findings: list[dict[str, Any]],
    scanned: list[str],
    roots: list[str] | None = None,
) -> dict[str, Any]:
    quarantine = [item for item in findings if item["classification"] == "asset_reference"]
    commentary = [item for item in findings if item["classification"] == "commentary"]
    by_rule: dict[str, int] = {}
    for item in quarantine:
        for rule in item["rules"]:
            by_rule[rule] = by_rule.get(rule, 0) + 1
    return {
        "schema_version": 1,
        "audited_on": date.today().isoformat(),
        "scanners_run": sorted(scanned),
        # An empty result means nothing was found *here*; it never means a root was clean when it
        # was not scanned at all. Cluster storage must be scanned on the cluster with --root.
        "filesystem_roots_scanned": sorted(roots or []),
        "rules": sorted({*INCOMPATIBLE_PATTERNS, *FILESYSTEM_ONLY_PATTERNS}),
        "totals": {
            "findings": len(findings),
            "policy_references": len(findings) - len(quarantine) - len(commentary),
            "commentary": len(commentary),
            "requiring_quarantine": len(quarantine),
            "by_rule": dict(sorted(by_rule.items())),
        },
        "git_branches": summarize_git_branches(findings) if "git" in scanned else [],
        "requiring_quarantine": sorted(
            quarantine, key=lambda item: (item["source"], json.dumps(item, sort_keys=True))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the JSON inventory report")
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="extra filesystem root to scan (repeatable; use for cluster shared storage)",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="entity/project to scan for artifacts and run configs (requires W&B read access)",
    )
    args = parser.parse_args()

    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    roots: list[Path] = []
    if not args.skip_git:
        findings.extend(scan_git())
        scanned.append("git")
    if not args.skip_local:
        roots = _local_roots(args.root)
        findings.extend(scan_local(roots))
        scanned.append("filesystem")
    if args.wandb_project:
        findings.extend(scan_wandb(args.wandb_project))
        scanned.append("wandb")

    report = build_report(findings, scanned, [str(root) for root in roots])
    if args.write:
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {REPORT_PATH}")
    else:
        print(json.dumps(report["totals"], indent=2, sort_keys=True))

    for branch in report["git_branches"]:
        merged = "merged" if branch["merged_into_main"] else "UNMERGED"
        print(
            f"  [git] {branch['ref']}: {branch['files']} files, "
            f"{branch['matches']} matches, {','.join(branch['rules'])} ({merged})"
        )
    for item in report["requiring_quarantine"]:
        if item["source"] == "git":
            continue
        location = item.get("root") or item.get("project", "")
        target = item.get("path") or item.get("artifact") or item.get("run")
        print(f"  [{item['source']}] {location}: {target} ({','.join(item['rules'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
