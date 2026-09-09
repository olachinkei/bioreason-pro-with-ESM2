"""Unit tests for scripts/fetch_interpro_annotations.py's pure TSV/parsing helpers (no network).

Covers a real bug found while running the Phase 1 InterPro batch job: `format_domains` joins multiple
domains with a real newline, and writing that directly into a TSV row split one protein's record
across several physical lines — a 1,496-protein run produced 5,362 "unique protein ids" once
multi-domain rows fragmented, because every continuation line was read back as its own bogus row.
"""

from __future__ import annotations

import csv
import io
import sys

sys.path.insert(0, "scripts")

from fetch_interpro_annotations import (  # noqa: E402
    escape_newlines,
    format_domains,
    parse_tsv,
    unescape_newlines,
)

MULTI_DOMAIN_TSV = "\n".join([
    "query\tmd5\t123\tPfam\tPF00001\t7tm_1\t10\t50\t1.2E-10\tT\t01-01-2026\tIPR000276\tGPCR family\t-\t-",
    "query\tmd5\t123\tPfam\tPF00002\t-\t70\t90\t-\tT\t01-01-2026\tIPR000277\tOther family\t-\t-",
])


def test_format_domains_joins_multiple_domains_with_a_real_newline():
    domains = parse_tsv(MULTI_DOMAIN_TSV)
    assert len(domains) == 2
    formatted = format_domains(domains)
    assert formatted == (
        "- IPR000276: GPCR family [10-50]\n- IPR000277: Other family [70-90]"
    )


def test_escape_unescape_newlines_round_trips():
    formatted = "- IPR000276: GPCR family [10-50]\n- IPR000277: Other family [70-90]"
    escaped = escape_newlines(formatted)
    assert "\n" not in escaped
    assert unescape_newlines(escaped) == formatted


def test_escaped_multi_domain_row_survives_a_tsv_round_trip():
    """The regression this bug caused: writing the RAW (un-escaped) formatted string breaks the next
    row's column alignment when read back with csv.DictReader. Escaping first must not."""
    pid = "A4IG66"
    formatted = "- IPR018617: Ima1, N-terminal domain [45-168]\n- IPR018861: Transmembrane [188-388]"
    next_pid, next_formatted = "A2AM05", "- IPR038810: Centlein [96-1396]"

    tsv_text = "protein_id\tinterpro_formatted\n"
    tsv_text += f"{pid}\t{escape_newlines(formatted)}\n"
    tsv_text += f"{next_pid}\t{escape_newlines(next_formatted)}\n"

    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    rows = list(reader)
    assert [r["protein_id"] for r in rows] == [pid, next_pid]  # exactly 2 rows, not 3+
    assert unescape_newlines(rows[0]["interpro_formatted"]) == formatted
    assert unescape_newlines(rows[1]["interpro_formatted"]) == next_formatted
