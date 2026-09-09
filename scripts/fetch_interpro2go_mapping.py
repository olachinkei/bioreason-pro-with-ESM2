#!/usr/bin/env python3
"""Fetch and verify the pinned GO Consortium `interpro2go` mapping (CC-BY-4.0).

Mirrors scripts/fetch_approved_reference_data.py's fail-closed checksum pattern, but for a single
flat file at a persistent URL rather than an archive member.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.license_policy import load_approved_assets  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    spec = load_approved_assets()["reference_data"]["interpro2go_mapping"]
    destination = ROOT / spec["file"]["path"]
    if destination.exists():
        size = destination.stat().st_size
        digest = _sha256(destination)
        if size == spec["file"]["size"] and digest == spec["file"]["sha256"]:
            print(f"{destination} already matches the pinned checksum.")
            return 0
        raise SystemExit(
            f"{destination} exists but does not match the pinned checksum "
            f"(expected size={spec['file']['size']} sha256={spec['file']['sha256']}, "
            f"got size={size} sha256={digest}); remove it before re-fetching."
        )

    print(f"Downloading {spec['source']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # obolibrary.org 403s the default urllib user-agent (no such restriction hit with a browser-like one).
    request = urllib.request.Request(spec["source"], headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as out:
        out.write(response.read())
    size = destination.stat().st_size
    digest = _sha256(destination)
    if size != spec["file"]["size"] or digest != spec["file"]["sha256"]:
        destination.unlink()
        raise SystemExit(
            f"Downloaded file failed verification: expected size={spec['file']['size']} "
            f"sha256={spec['file']['sha256']}, got size={size} sha256={digest}. This usually means "
            "the GO Consortium has published a newer interpro2go release — re-pin the manifest "
            "deliberately rather than silently accepting a different file."
        )
    print(f"Installed verified interpro2go mapping. Attribution: {spec['attribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
