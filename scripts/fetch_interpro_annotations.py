#!/usr/bin/env python3
"""Fetch InterPro domain annotations for the Phase 1 temporal dev set via EBI's public InterProScan5
REST API (https://www.ebi.ac.uk/Tools/services/rest/iprscan5) — the same service upstream's
`interpro_api.py` calls (ADR-028: no resource-specific licensing restriction found in that client).

This is a long-running external batch job by construction: EBI's public dispatcher queues jobs behind
other users' work, and per-job wait times upstream observed varied from ~90s to ~600s even for jobs
submitted together. For ~1,500 proteins this is many hours regardless of how politely it is run.
plan.md Phase 1 explicitly permits shipping the dev set sequence-only while this runs in the
background — do not block on it finishing.

Resumable: writes each completed protein's result to --out immediately (append + flush) and skips ids
already present in --out on restart, so an interrupted run picks up where it left off rather than
resubmitting jobs that already finished.

Usage:
    python scripts/fetch_interpro_annotations.py --email you@example.org
    python scripts/fetch_interpro_annotations.py --email you@example.org --limit 20   # smoke a subset
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
DEV_SET_PATH = ROOT / "data" / "cafa_no_knowledge_dev_set.jsonl"
DEFAULT_OUT = ROOT / "data" / "cafa_no_knowledge_interpro.tsv"

IPRSCAN_API_URL = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
REQUEST_TIMEOUT = 90      # seconds per HTTP request
POLL_TIMEOUT = 1800       # seconds to wait for one InterProScan job (matches upstream's bound)
POLL_INTERVAL = 10        # seconds between status polls
REQUEST_RETRY_SLEEPS = (5, 15, 30)  # transient SSL/handshake blips observed in practice on this
                                     # network path; upstream's client had no retry at all

TSV_COLS = [
    "protein", "md5", "length", "analysis", "signature_acc", "signature_desc",
    "start", "end", "score", "status", "date", "interpro_id", "interpro_desc", "go_terms", "pathway",
]

_write_lock = Lock()


def _request(url: str, data: bytes | None = None) -> str:
    req = urllib.request.Request(url, data=data)
    for sleep_s in (*REQUEST_RETRY_SLEEPS, None):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8")
        except Exception:  # noqa: BLE001 - retry any transient network error; re-raise on the last attempt
            if sleep_s is None:
                raise
            time.sleep(sleep_s)


def submit_job(sequence: str, email: str) -> str:
    payload = urllib.parse.urlencode({"email": email, "sequence": sequence, "stype": "p"}).encode()
    return _request(f"{IPRSCAN_API_URL}/run", data=payload).strip()


def poll_job(job_id: str) -> None:
    """Block until `job_id` reaches FINISHED, or raise on failure/timeout."""
    deadline = time.time() + POLL_TIMEOUT
    status = "UNKNOWN"
    while time.time() < deadline:
        status = _request(f"{IPRSCAN_API_URL}/status/{job_id}").strip()
        if status == "FINISHED":
            return
        if status in ("FAILURE", "ERROR", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job_id} failed with status: {status}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"InterProScan job {job_id} still {status} after {POLL_TIMEOUT}s")


def parse_tsv(text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    rows = [r for r in reader if r and not r[0].startswith("#")]
    by_domain: dict[tuple[str, str], dict] = {}
    for row in rows:
        cols = dict(zip(TSV_COLS, row))
        ipr_id = cols.get("interpro_id")
        if not ipr_id or ipr_id == "-":
            continue
        key = (ipr_id, cols.get("interpro_desc", ""))
        start, end = int(cols["start"]), int(cols["end"])
        entry = by_domain.setdefault(key, {"start": start, "end": end})
        entry["start"] = min(entry["start"], start)
        entry["end"] = max(entry["end"], end)
    return [
        {"interpro_id": ipr_id, "entry_name": desc, "start": v["start"], "end": v["end"]}
        for (ipr_id, desc), v in sorted(by_domain.items())
    ]


def escape_newlines(text: str) -> str:
    """`\\n` literal, so a multi-domain value stays on one TSV line (see fetch_one's caller)."""
    return text.replace("\n", "\\n")


def unescape_newlines(text: str) -> str:
    return text.replace("\\n", "\n")


def format_domains(domains: list[dict]) -> str:
    if not domains:
        return ""
    return "\n".join(f"- {d['interpro_id']}: {d['entry_name']} [{d['start']}-{d['end']}]" for d in domains)


def fetch_one(protein_id: str, sequence: str, email: str) -> tuple[str, str]:
    job_id = submit_job(sequence, email)
    poll_job(job_id)
    text = _request(f"{IPRSCAN_API_URL}/result/{job_id}/tsv")
    return protein_id, format_domains(parse_tsv(text))


def _load_sequences(limit: int | None) -> list[tuple[str, str]]:
    if not DEV_SET_PATH.exists():
        raise SystemExit(f"{DEV_SET_PATH} does not exist; run scripts/build_cafa_no_knowledge_dev_set.py first")
    rows = []
    with DEV_SET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            rows.append((rec["protein_id"], rec["sequence"]))
    return rows[:limit] if limit else rows


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["protein_id"] for row in reader}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="required by EBI for job submission")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--submit-stagger-seconds", type=float, default=2.0,
                         help="delay between successive job submissions, to stay a polite client")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N proteins (smoke)")
    args = parser.parse_args()

    all_rows = _load_sequences(args.limit)
    done = _already_done(args.out)
    pending = [(pid, seq) for pid, seq in all_rows if pid not in done]
    print(f"[interpro] {len(done)} already done, {len(pending)} pending out of {len(all_rows)} total",
          file=sys.stderr)

    is_new_file = not args.out.exists()
    out_handle = args.out.open("a", encoding="utf-8")
    if is_new_file:
        out_handle.write("protein_id\tinterpro_formatted\n")
        out_handle.flush()

    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for pid, seq in pending:
            futures[pool.submit(fetch_one, pid, seq, args.email)] = pid
            time.sleep(args.submit_stagger_seconds / args.concurrency)
        for future in as_completed(futures):
            pid = futures[future]
            try:
                _, formatted = future.result()
            except Exception as exc:  # noqa: BLE001 - record and continue; a lone job's failure
                                       # (EBI queue timeout, transient 5xx) must not abort the batch
                print(f"[interpro] {pid}: FAILED — {exc}", file=sys.stderr)
                failures.append(pid)
                continue
            with _write_lock:
                # format_domains joins multiple domains with a real newline, which would otherwise
                # split one TSV row across several physical lines and corrupt every later row's
                # column alignment (found in practice: a 1,496-row run produced 5,362 "unique ids"
                # once multi-domain rows fragmented). Escape before writing, unescape on read
                # (escape_newlines/unescape_newlines below) — Python 3.10 disallows a backslash
                # inside an f-string expression, hence the helper rather than inline .replace.
                out_handle.write(f"{pid}\t{escape_newlines(formatted)}\n")
                out_handle.flush()
            print(f"[interpro] {pid}: {formatted.count(chr(10)) + 1 if formatted else 0} domain(s)",
                  file=sys.stderr)

    out_handle.close()
    print(f"[interpro] done: {len(pending) - len(failures)} succeeded, {len(failures)} failed", file=sys.stderr)
    if failures:
        print(f"[interpro] failed ids (rerun this script to retry — it skips completed ones): {failures}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
