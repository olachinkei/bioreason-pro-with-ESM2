#!/usr/bin/env python3
"""Build plan.md Phase 1's temporal development set: CAFA's no-knowledge targets minus the sealed
holdout, with UniProt sequences and (once available) InterPro context.

Prerequisite: `python scripts/fetch_approved_reference_data.py` (pulls
`data/eval_terms_no_knowledge_2025_03.tsv` and `data/known_t0.tsv` from the pinned CAFA5 Zenodo
bundle; this script only reads them).

Writes:
  - data/cafa_no_knowledge_ids.txt        — sorted protein ids, one per line (like public_holdout_ids.txt)
  - data/cafa_no_knowledge_dev_set.jsonl  — {protein_id, sequence, go_mf, go_bp, go_cc, organism,
                                             interpro_formatted?} one JSON object per line

Does NOT check disjointness against the training corpus (that needs streaming two multi-hundred-MB HF
datasets) — see scripts/verify_cafa_no_knowledge_disjoint_from_corpus.py for that, run separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioreason_pro.dev_set import build_dev_ids, parse_no_knowledge_terms  # noqa: E402
from bioreason_pro.license_policy import validate_reference_assets  # noqa: E402

NO_KNOWLEDGE_TSV = ROOT / "data" / "eval_terms_no_knowledge_2025_03.tsv"
HOLDOUT_IDS = ROOT / "data" / "public_holdout_ids.txt"
IDS_OUT = ROOT / "data" / "cafa_no_knowledge_ids.txt"
DEV_SET_OUT = ROOT / "data" / "cafa_no_knowledge_dev_set.jsonl"
INTERPRO_TSV = ROOT / "data" / "cafa_no_knowledge_interpro.tsv"  # optional, from fetch_interpro_annotations.py

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
BATCH_SIZE = 50   # smaller batches: observed single-batch latency at 90/batch was ~40s and sometimes
                  # exceeded a 60s timeout outright, so shrink the request instead of just the timeout
REQUEST_TIMEOUT = 120
RETRY_SLEEPS = (5, 15, 30)  # backoff; last attempt still counts toward the batch's own failure
SEQUENCE_CACHE = Path("/tmp/cafa_no_knowledge_sequence_cache.json")  # scratch, not repo state; lets a
                                                                     # stalled late batch resume instead
                                                                     # of re-fetching everything (observed
                                                                     # in practice: a single batch can hang
                                                                     # past every timeout/retry)
ORGANISM_CACHE = Path("/tmp/cafa_no_knowledge_organism_cache.json")  # same resumability rationale


def _read_no_knowledge_ids_and_terms() -> tuple[set[str], dict[str, dict[str, set[str]]]]:
    with NO_KNOWLEDGE_TSV.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        terms = parse_no_knowledge_terms(reader)
    return set(terms), terms


def _read_holdout_ids() -> set[str]:
    with HOLDOUT_IDS.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _fetch_sequences(ids: list[str]) -> dict[str, str]:
    """UniProtKB REST `stream` endpoint, batched FASTA, CC-BY-4.0 (approved_assets.json column_provenance).

    Resumable via SEQUENCE_CACHE: a batch already cached from a prior (possibly stalled/killed) run is
    skipped rather than re-fetched, since re-running the whole 1,496-id fetch from scratch after one
    late batch hangs is a real cost this project's own method rules ask not to accept by default.
    """
    sequences: dict[str, str] = {}
    if SEQUENCE_CACHE.exists():
        sequences = json.loads(SEQUENCE_CACHE.read_text(encoding="utf-8"))
        print(f"[uniprot] resuming from cache: {len(sequences)} sequences already fetched", file=sys.stderr)
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start:start + BATCH_SIZE]
        if all(pid in sequences for pid in batch):
            continue
        query = "+OR+".join(f"accession:{pid}" for pid in batch)
        url = f"{UNIPROT_STREAM_URL}?query={query}&format=fasta"
        for attempt, sleep_s in enumerate((*RETRY_SLEEPS, None)):
            try:
                with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
                    text = resp.read().decode("utf-8")
                break
            except Exception as exc:  # noqa: BLE001 - retry on any transient network error
                if sleep_s is None:
                    raise RuntimeError(f"UniProt fetch failed for batch starting at {start}: {exc}") from exc
                print(f"[uniprot] batch at {start} failed ({exc}); retrying in {sleep_s}s "
                      f"(attempt {attempt + 1}/{len(RETRY_SLEEPS)})", file=sys.stderr)
                time.sleep(sleep_s)
        pid = None
        parts: list[str] = []
        for line in text.splitlines():
            if line.startswith(">"):
                if pid is not None:
                    sequences[pid] = "".join(parts)
                header_id = line[1:].split("|")
                pid = header_id[1] if len(header_id) >= 2 else line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
        if pid is not None:
            sequences[pid] = "".join(parts)
        SEQUENCE_CACHE.write_text(json.dumps(sequences), encoding="utf-8")
        print(f"[uniprot] fetched {len(sequences)}/{len(ids)} sequences so far", file=sys.stderr)
    return sequences


def _fetch_organisms(ids: list[str]) -> dict[str, str]:
    """UniProtKB REST `stream` endpoint, TSV `organism_name` field (plan.md Phase 5).

    Verified against the training corpus's own `organism` column before writing this: querying this
    same endpoint for Q5RK27/P37592 returns 'Rattus norvegicus (Rat)'/'Salmonella typhimurium (strain
    LT2 / SGSC1412 / ATCC 700720)' -- byte-identical to what those two proteins' corpus rows carry --
    so the dev set's rendering matches training's prompt shape exactly, the same standard
    CONTEXT_COLUMNS already holds itself to.
    """
    organisms: dict[str, str] = {}
    if ORGANISM_CACHE.exists():
        organisms = json.loads(ORGANISM_CACHE.read_text(encoding="utf-8"))
        print(f"[uniprot] resuming organism fetch from cache: {len(organisms)} already fetched",
              file=sys.stderr)
    for start in range(0, len(ids), BATCH_SIZE):
        batch = ids[start:start + BATCH_SIZE]
        if all(pid in organisms for pid in batch):
            continue
        query = "+OR+".join(f"accession:{pid}" for pid in batch)
        url = f"{UNIPROT_STREAM_URL}?query={query}&format=tsv&fields=accession,organism_name"
        for attempt, sleep_s in enumerate((*RETRY_SLEEPS, None)):
            try:
                with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
                    text = resp.read().decode("utf-8")
                break
            except Exception as exc:  # noqa: BLE001 - retry on any transient network error
                if sleep_s is None:
                    raise RuntimeError(f"UniProt organism fetch failed for batch at {start}: {exc}") from exc
                print(f"[uniprot] organism batch at {start} failed ({exc}); retrying in {sleep_s}s "
                      f"(attempt {attempt + 1}/{len(RETRY_SLEEPS)})", file=sys.stderr)
                time.sleep(sleep_s)
        lines = text.splitlines()[1:]  # header row: "Entry\tOrganism"
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                organisms[parts[0]] = parts[1]
        ORGANISM_CACHE.write_text(json.dumps(organisms), encoding="utf-8")
        print(f"[uniprot] fetched {len(organisms)}/{len(ids)} organisms so far", file=sys.stderr)
    return organisms


def _read_interpro(path: Path) -> dict[str, str]:
    """Reads scripts/fetch_interpro_annotations.py's TSV, unescaping the `\\n` it writes in place of
    a real newline (a multi-domain value would otherwise split one row across several TSV lines)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[row["protein_id"]] = row["interpro_formatted"].replace("\\n", "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-count", type=int, default=1496)
    args = parser.parse_args()

    validate_reference_assets(use="evaluation")  # fail closed if the pinned bundle isn't present/verified

    no_knowledge_ids, terms_by_protein = _read_no_knowledge_ids_and_terms()
    holdout_ids = _read_holdout_ids()
    dev_ids = build_dev_ids(no_knowledge_ids, holdout_ids, expected_count=args.expected_count)
    print(f"[dev-set] {len(no_knowledge_ids)} CAFA no-knowledge ids, "
          f"{len(no_knowledge_ids & holdout_ids)} overlap the sealed holdout, "
          f"{len(dev_ids)} remain")

    sorted_ids = sorted(dev_ids)
    IDS_OUT.write_text("\n".join(sorted_ids) + "\n", encoding="utf-8")

    sequences = _fetch_sequences(sorted_ids)
    missing = [pid for pid in sorted_ids if pid not in sequences]
    if missing:
        raise SystemExit(
            f"UniProt returned no sequence for {len(missing)} ids (e.g. {missing[:5]}); "
            "the accession may be demerged/retired — resolve before shipping the dev set"
        )
    organisms = _fetch_organisms(sorted_ids)  # optional field (bioreason_pro.data_contract.ORGANISM_COLUMN
                                               # renders "not available" itself), so a gap here is logged,
                                               # not fatal the way a missing sequence is

    interpro = _read_interpro(INTERPRO_TSV)
    retrieval_date = date.today().isoformat()  # UniProt is versioned; this pins what was fetched
    with DEV_SET_OUT.open("w", encoding="utf-8") as handle:
        for pid in sorted_ids:
            record = {
                "protein_id": pid,
                "sequence": sequences[pid],
                "go_mf": sorted(terms_by_protein.get(pid, {}).get("go_mf", set())),
                "go_bp": sorted(terms_by_protein.get(pid, {}).get("go_bp", set())),
                "go_cc": sorted(terms_by_protein.get(pid, {}).get("go_cc", set())),
                "organism": organisms.get(pid, ""),
                "sequence_retrieval_date": retrieval_date,
            }
            if pid in interpro:
                record["interpro_formatted"] = interpro[pid]
            handle.write(json.dumps(record) + "\n")

    print(f"[dev-set] wrote {len(sorted_ids)} ids to {IDS_OUT}")
    print(f"[dev-set] wrote {len(sorted_ids)} records to {DEV_SET_OUT}")
    print(f"[dev-set] InterPro context present for {len(interpro)}/{len(sorted_ids)} proteins "
          f"({'none — sequence-only, as declared in eval_targets/cafa_no_knowledge.py' if not interpro else 'partial/full'})")
    print(f"[dev-set] organism present for {len(organisms)}/{len(sorted_ids)} proteins "
          f"(plan.md Phase 5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
