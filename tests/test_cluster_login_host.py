"""Only one CoreWeave login host is real for this project.

`sunk.cwb607-trainingsss.coreweave.app` (166.19.38.113) is the cluster with H100 nodes.
`sunk.cwb607-training.coreweave.app` (166.19.16.193) — same name minus the `sss` — is a different
cluster with no nodes registered, where `sinfo` shows 0 nodes and every job sits in
`PENDING (PartitionConfig)` forever. The two are one typo apart and the failure is silent: jobs are
accepted and simply never start, which reads as a queue backlog rather than a wrong hostname.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORRECT_HOST = "sunk.cwb607-trainingsss.coreweave.app"
WRONG_HOST = "sunk.cwb607-training.coreweave.app"

TRACKED_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sbatch", ".tmpl", ".sh", ".json"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "outputs", "wandb", "senpai"}


def _tracked_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TRACKED_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


# These name the wrong host in order to forbid it.
ALLOWED_TO_MENTION = {"tests/test_cluster_login_host.py", "RUNBOOK.md"}


def test_no_file_points_at_the_nodeless_cluster():
    offenders = []
    for path in _tracked_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if WRONG_HOST not in text:
            continue
        relative = str(path.relative_to(ROOT))
        if relative not in ALLOWED_TO_MENTION:
            offenders.append(relative)
    assert not offenders, (
        f"{offenders} reference {WRONG_HOST}; use {CORRECT_HOST} (note the 'sss')"
    )


def test_runbook_documents_the_correct_host_and_the_trap():
    runbook = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    assert CORRECT_HOST in runbook
    assert "note the `sss`" in runbook.lower() or "note the sss" in runbook.lower()
    assert "PartitionConfig" in runbook


def test_runbook_documents_the_best_known_configuration():
    """The winning combination is entirely opt-in, so defaults reproduce the original recipe.

    Without this section a reader following the RUNBOOK gets ~0.112, not ~0.310, and nothing tells
    them which flags they are missing.
    """
    runbook = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")
    for flag in (
        "SENPAI_TARGET_VARIANT=leaf_only",
        "SENPAI_REWARD_VARIANT=aspect_mean",
        "RL_NUM_GENERATIONS=16",
        "RL_MAX_COMPLETION_LENGTH=64",
    ):
        assert flag in runbook, f"RUNBOOK must document {flag}"
