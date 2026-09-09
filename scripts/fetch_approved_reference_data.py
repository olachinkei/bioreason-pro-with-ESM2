#!/usr/bin/env python3
"""Fetch and verify the approved CAFA evaluation ontology and IA weights."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.license_policy import (  # noqa: E402
    load_approved_assets,
    validate_reference_assets,
)


def _digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def _copy_member(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise RuntimeError(f"{member_name}: expected a regular file")
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"{member_name}: could not read archive member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing file only when it does not match the approved checksum",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    bundle = load_approved_assets()["reference_data"]["cafa5_evaluation_bundle"]
    destinations = {root / relative: spec for relative, spec in bundle["files"].items()}
    mismatched = []
    pending = []
    for destination, spec in destinations.items():
        if not destination.exists():
            pending.append((destination, spec))
            continue
        size, _, sha256 = _digests(destination)
        if size == spec["size"] and sha256 == spec["sha256"]:
            continue
        mismatched.append(destination)
        pending.append((destination, spec))
    if mismatched and not args.force:
        paths = "\n".join(f"- {path}" for path in mismatched)
        raise SystemExit(
            "Refusing to replace mismatched local files:\n"
            f"{paths}\nReview them, then rerun with --force if replacement is intended."
        )
    if not pending:
        validate_reference_assets(root, use="evaluation")
        print("Approved GO ontology and IA weights already match the pinned bundle.")
        return 0

    with tempfile.TemporaryDirectory(prefix="bioreasonpro-reference-") as temp_dir:
        archive_path = Path(temp_dir) / "full_evaluation.tar.gz"
        print(f"Downloading {bundle['archive_url']}")
        urllib.request.urlretrieve(bundle["archive_url"], archive_path)
        size, md5, sha256 = _digests(archive_path)
        expected = (bundle["archive_size"], bundle["archive_md5"], bundle["archive_sha256"])
        if (size, md5, sha256) != expected:
            raise SystemExit(
                "Downloaded archive failed verification: "
                f"got size={size} md5={md5} sha256={sha256}"
            )

        with tarfile.open(archive_path, "r:gz") as archive:
            for destination, spec in pending:
                staged = Path(temp_dir) / destination.name
                _copy_member(archive, spec["archive_path"], staged)
                staged_size, _, staged_sha256 = _digests(staged)
                if staged_size != spec["size"] or staged_sha256 != spec["sha256"]:
                    raise SystemExit(f"{spec['archive_path']}: extracted file failed verification")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)

    validate_reference_assets(root, use="evaluation")
    print(f"Installed verified reference data. Attribution: {bundle['attribution']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
