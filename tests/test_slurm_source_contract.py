"""Static contracts for reproducible CoreWeave launchers."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    ROOT / "slurm" / "sft_smoke.sbatch",
    ROOT / "slurm" / "sft.sbatch",
    ROOT / "slurm" / "rl_grpo_smoke.sbatch",
    ROOT / "slurm" / "rl_grpo.sbatch",
)


def test_training_launchers_require_an_immutable_source_revision():
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert 'SOURCE_REVISION:?' in text, launcher
        assert "^[0-9a-f]{40}$" in text, launcher
        assert 'checkout -q --detach FETCH_HEAD' in text, launcher
        assert 'fetch -q "$SOURCE_BUNDLE" "$SOURCE_REVISION"' in text, launcher
        assert 'actual_revision="$(git -C "$REPO_DIR" rev-parse HEAD)"' in text, launcher
        assert 'BIOREASON_SOURCE_REVISION="$SOURCE_REVISION"' in text, launcher
        assert "SOURCE_BRANCH" not in text, launcher


def test_training_launchers_do_not_require_github_credentials_on_compute_nodes():
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert '[[ -r "$SOURCE_BUNDLE" ]]' in text, launcher
        assert "SOURCE_REPOSITORY_URL" in text, launcher


def test_training_launchers_request_typed_h100_gres():
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "#SBATCH --gres=gpu:h100:" in text, launcher


def test_container_import_uses_compute_local_temporary_storage():
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert 'export BIOREASON_JOB_TMP=' in text, launcher
        assert "unset TMPDIR" in text, launcher
        assert 'export TMPDIR="$BIOREASON_JOB_TMP"' in text, launcher
        assert text.index("unset TMPDIR") < text.index("srun --container-image"), launcher


def test_training_launchers_have_valid_bash_syntax():
    for launcher in LAUNCHERS:
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_rl_launchers_retry_transient_reference_downloads():
    for name in ("rl_grpo_smoke.sbatch", "rl_grpo.sbatch"):
        text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert "for attempt in 1 2 3" in text
        assert "approved reference data download failed after 3 attempts" in text
        assert "BIOREASON_REFERENCE_DATA_ROOT" in text
        assert text.index("BIOREASON_REFERENCE_DATA_ROOT") < text.index(
            "python scripts/fetch_approved_reference_data.py"
        )
