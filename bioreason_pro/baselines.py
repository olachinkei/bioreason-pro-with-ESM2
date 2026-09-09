"""bioreason_pro.baselines — pure helpers for plan.md Phase 2's zero-parameter reference baselines.

Rule 13: this project ran eleven phases before measuring a free baseline, and when one was finally
computed (ADR-022) it beat the trained model on the split then in use. Both baselines here score
through the same `eval.score_generations` path every model arm uses, not a reward-style proxy — they
just need a synthetic "generated_response" string containing the predicted GO ids, since
`extract_go_terms` only regex-scans text for `GO:#######` patterns and does not care where they came
from, so the metric is identical to every other row in the ledger.
"""

from __future__ import annotations

import re
from collections import Counter

_IPR_RE = re.compile(r"IPR\d{6}")
# GO Consortium interpro2go line shape: "InterPro:IPR000003 <name> > GO:<name> ; GO:0003677"
_INTERPRO2GO_LINE_RE = re.compile(r"^InterPro:(IPR\d{6})\b.*?;\s*(GO:\d{7})\s*$")


def parse_interpro2go_mapping(lines) -> dict[str, set[str]]:
    """Parse the GO Consortium `interpro2go` file into `{IPR_id: {GO_id, ...}}`.

    `lines`: an iterable of raw text lines (e.g. an open file handle). `!`-prefixed and blank lines
    are the file's own header/comment convention and are skipped; anything else that doesn't match
    the expected shape is a real parsing failure, not silently dropped.
    """
    mapping: dict[str, set[str]] = {}
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("!"):
            continue
        match = _INTERPRO2GO_LINE_RE.match(line)
        if not match:
            raise ValueError(f"unrecognized interpro2go line: {line!r}")
        ipr_id, go_id = match.groups()
        mapping.setdefault(ipr_id, set()).add(go_id)
    return mapping


def extract_interpro_ids(interpro_formatted: str | None) -> set[str]:
    """IPR ids out of a WangLab-formatted InterPro block. None/empty/'not available' -> empty set."""
    if not interpro_formatted:
        return set()
    return set(_IPR_RE.findall(interpro_formatted))


def interpro2go_predict(interpro_ids: set[str], mapping: dict[str, set[str]]) -> set[str]:
    """Union of GO terms the given InterPro ids map to. Zero-parameter: no protein-specific choice,
    only the fixed public lookup table."""
    predicted: set[str] = set()
    for ipr_id in interpro_ids:
        predicted |= mapping.get(ipr_id, set())
    return predicted


def label_prior_terms(term_counts: dict[str, Counter], top_n: dict[str, int]) -> dict[str, set[str]]:
    """The `top_n[aspect]` most frequent GO terms per aspect, ignoring any specific protein.

    `term_counts`: `{"go_mf": Counter({GO_id: count, ...}), "go_bp": ..., "go_cc": ...}` over the
    training corpus. Returns the SAME fixed set for every protein regardless of `top_n` — that is the
    point of this baseline: it isolates a score that reflects the label distribution alone, with zero
    protein-specific signal, so a model beating it is doing more than reciting the prior.
    """
    return {
        aspect: {term for term, _ in counts.most_common(top_n.get(aspect, 0))}
        for aspect, counts in term_counts.items()
    }


def as_generated_response(terms: set[str]) -> str:
    """Render a predicted term set as text `eval.score_generations`'s `extract_go_terms` can parse.

    `extract_go_terms` just regex-scans for `GO:#######` anywhere in the text, so any separator
    works; space-joined sorted ids keep this deterministic and legible in a stored predictions file.
    """
    return " ".join(sorted(terms))
