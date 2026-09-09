"""The submit helper must close the two footguns that already cost scheduled jobs.

1. Submitting after a merge without staging the revision's bundle. The sbatch scripts fetch their
   source from a bundle on shared storage and nothing checks it exists, so the job is scheduled,
   starts, and dies two seconds later on "SOURCE_BUNDLE is not readable" (job 635).
2. Passing a comma-containing value through `sbatch --export=K=V`, where commas separate
   assignments, so `PROMPT_COMPLETION_LENGTHS=64,128,256` silently becomes `...=64` plus three bogus
   entries (job 628).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "submit_slurm.sh"


def test_helper_exists_and_is_executable():
    assert HELPER.is_file()
    assert os.stat(HELPER).st_mode & stat.S_IXUSR, "helper must be executable"


def test_helper_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(HELPER)], capture_output=True).returncode == 0


def test_helper_stages_and_verifies_the_bundle_before_submitting():
    text = HELPER.read_text(encoding="utf-8")
    build = text.index("git bundle create")
    verify = text.index("verifying the bundle fetches")
    submit = text.index("sbatch $SBATCH_ARGS --export=ALL")
    assert build < verify < submit, "must build, then verify, then submit"
    # The verification must prove the revision, not merely that a file exists.
    assert "rev-parse HEAD" in text
    assert "bundle revision mismatch" in text


def test_helper_refuses_a_dirty_tree():
    """A dirty tree means the bundle would not contain what the author is testing."""
    text = HELPER.read_text(encoding="utf-8")
    assert "git status --porcelain" in text
    assert "working tree is dirty" in text


def test_helper_never_passes_values_through_sbatch_export_assignments():
    import re

    text = HELPER.read_text(encoding="utf-8")
    # Values are exported in the submitting shell and carried by --export=ALL. Check the real
    # invariant on executable lines only: comments are allowed to name the broken form in order to
    # explain why it is avoided.
    code = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    for line in code:
        for found in re.findall(r"--export=(\S+)", line):
            assert found == "ALL", f"only --export=ALL is safe, found --export={found}"
    assert "--export=ALL" in "\n".join(code)
    # Only real invocations: a line whose first token is `sbatch`. Usage text mentions
    # "<sbatch-path>" and variables are named SBATCH_PATH, neither of which is a call.
    invocations = [
        line.strip() for line in text.splitlines() if line.strip().startswith("sbatch ")
    ]
    assert invocations, "expected at least one sbatch invocation"
    for call in invocations:
        assert "--export=ALL" in call, f"sbatch call must use --export=ALL: {call!r}"
