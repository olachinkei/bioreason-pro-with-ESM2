"""`timeout` intervals in sbatch scripts must be a form GNU coreutils accepts.

`timeout 3h30m` is not valid: coreutils takes ONE number with an optional s/m/h/d suffix, so a
compound duration exits 125 immediately. That is expensive here — the job allocates a GPU, runs
`uv sync`, downloads the reference bundle, and only then dies, so the mistake burns an allocation
and reads as an infrastructure failure rather than a typo.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# One number (optionally fractional), optionally suffixed with s/m/h/d. Nothing more.
VALID_INTERVAL = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
TIMEOUT_CALL = re.compile(r"\btimeout\b((?:\s+--?[\w-]+(?:=\S+)?)*)\s+(\S+)")


def _sbatch_files():
    return sorted((ROOT / "slurm").rglob("*.sbatch")) + sorted((ROOT / "slurm").rglob("*.tmpl"))


def test_every_sbatch_timeout_interval_is_valid_for_coreutils():
    offenders = []
    for path in _sbatch_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for _flags, interval in TIMEOUT_CALL.findall(line):
                if not VALID_INTERVAL.match(interval):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} -> {interval!r}")
    assert not offenders, (
        "GNU timeout takes a single number with an optional s/m/h/d suffix "
        f"(use 210m, not 3h30m): {offenders}"
    )


def test_the_pattern_would_have_caught_the_compound_duration():
    """Guard the guard: the regex must actually reject the form that failed job 620."""
    assert not VALID_INTERVAL.match("3h30m")
    assert not VALID_INTERVAL.match("1h30")
    for good in ("210m", "23h", "3.5h", "90", "30s", "1d"):
        assert VALID_INTERVAL.match(good), good


def test_timeout_calls_are_actually_found():
    """If the extraction regex silently matched nothing, the first test would pass vacuously."""
    found = [
        interval
        for path in _sbatch_files()
        for _flags, interval in TIMEOUT_CALL.findall(path.read_text(encoding="utf-8"))
    ]
    assert found, "no timeout invocations found in slurm/ — the extraction regex has drifted"
