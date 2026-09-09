"""`zoneinfo` must be able to resolve UTC wherever this runs.

weave.Evaluation resolves UTC through `zoneinfo`, which reads the system tz database. The container
image does not ship `/usr/share/zoneinfo`, and the `tzdata` wheel — zoneinfo's fallback — only
arrived transitively under a `python_full_version < "3.11"` marker, so on a newer interpreter the
Weave publish died with `ZoneInfoNotFoundError: 'No time zone found with key UTC'` (observed in job
629). The metric of record survived because publishing is best-effort, but the Evaluation was lost.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def test_utc_resolves_in_this_environment():
    assert ZoneInfo("UTC") is not None


def test_tzdata_is_an_unconditional_base_dependency():
    """A marker-gated tzdata is what let the container fall through the gap."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    entries = [d for d in deps if d.split(">=")[0].split("==")[0].strip() == "tzdata"]
    assert entries, f"tzdata must be a base dependency; got {deps}"
    for entry in entries:
        assert ";" not in entry, f"tzdata must not be marker-gated: {entry!r}"
