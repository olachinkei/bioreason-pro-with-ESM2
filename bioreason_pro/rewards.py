"""Composite RL reward for GRPO/GSPO (AUTHORED — one of the genuinely-missing pieces).

PROTECTED so it cannot be redefined in a PR to game the metric. Shared by --rl_algo grpo|gspo.

    r = r_fmax + lambda_fmt * r_format + lambda_len * r_conciseness

- r_fmax     (dominant): IA-weighted F1 of predicted vs ground-truth GO terms (ancestor-
             propagated) — a fast proxy for the threshold-swept cafaeval F_max used at eval time.
- r_format   (small): well-formed <think>…</think> + at least one predicted GO term.
- r_conciseness (small, capped): mild, bounded length penalty so it never dominates.

Conventions match the AUTHORITATIVE public eval (bioreason2 evals/cafa_evals.py):
- GO terms are GO:\\d{7} extracted from the WHOLE response (optionally only after </think>),
  NOT scoped to a <|GO_SUMMARY_START|> block (that block is not guaranteed to exist).
- Ground truth comes from the go_bp/go_mf/go_cc columns (lists, or stringified lists).
Reward weights / KL / clip-eps are flagged uncertainties to pin from the paper.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from math import isfinite

GO_RE = re.compile(r"GO:\d{7}")
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_GT_COLUMNS = ("go_mf", "go_bp", "go_cc")


@dataclass
class RewardWeights:
    # Local invention, not paper-derived (ADR-027): the paper's RL reward is a single weighted-F_max
    # term with no format/conciseness counterpart, and upstream ships no RL script to have pinned
    # these against anyway. Kept non-zero here only for union/aspect_mean/aspect_mean_specific's own
    # historical reproducibility; train.py's aspect_mean_reasoned (Phase 6) zeroes both explicitly.
    lambda_fmt: float = 0.1
    lambda_len: float = 0.05
    len_target_tokens: int = 512
    len_penalty_cap: float = 1.0
    # Penalty on GO ids the completion names when another named id already implies them. 0.0 keeps
    # the shipped reward exactly; the aspect_mean_specific variant raises it.
    lambda_spec: float = 0.0
    # --- reasoning-quality terms (ADR-014) -------------------------------------------------------
    # All default to 0.0, so every pre-existing variant keeps its exact arithmetic. The
    # `aspect_mean_reasoned` variant raises them.
    lambda_reason: float = 0.0      # weight on substance x faithfulness
    lambda_truncation: float = 0.0  # penalty for an answer the budget cut off mid-identifier
    # A trace shorter than this is not an explanation; one past the target earns no more credit.
    # Characters, not tokens, so the term is tokenizer-independent.
    reason_min_chars: int = 120
    reason_target_chars: int = 600


@dataclass(frozen=True)
class RewardComponents:
    total: float
    fmax: float
    format: float
    conciseness: float
    redundancy: float = 0.0
    substance: float = 0.0
    faithfulness: float = 0.0
    truncation: float = 0.0


def extract_go_terms(text: str, final_answer_only: bool = False) -> set[str]:
    """GO:####### ids from the whole response (or only after </think>). Mirrors cafa_evals."""
    if final_answer_only and "</think>" in text:
        text = text.split("</think>")[-1]
    return set(GO_RE.findall(text))


def r_format(completion: str) -> float:
    """0.5 for a well-formed <think>…</think>, +0.5 for >=1 predicted GO term (bounded [0,1])."""
    return 0.5 * bool(THINK.search(completion)) + 0.5 * bool(GO_RE.search(completion))


def r_conciseness(completion: str, w: RewardWeights) -> float:
    """Bounded, non-positive length penalty: 0 up to target, then linear down to -cap."""
    n_tokens = len(completion.split())
    if n_tokens <= w.len_target_tokens:
        return 0.0
    over = (n_tokens - w.len_target_tokens) / w.len_target_tokens
    return -min(over, w.len_penalty_cap)


def reasoning_text(completion: str) -> str:
    """The contents of the <think> block, or '' when there is none."""
    match = THINK.search(completion)
    return match.group(1).strip() if match else ""


def r_reasoning_substance(completion: str, w: RewardWeights) -> float:
    """0 for an empty trace, ramping to 1 as it reaches an explanatory length. Bounded [0,1].

    Phase 7-9 shipped a `Qwen3-4B-Thinking` backbone that emitted an EMPTY <think> block in 472 of
    472 stored rollouts, because nothing in the reward or the supervision ever asked for one. This
    is the term that asks. It saturates at `reason_target_chars` so it cannot be farmed by padding;
    `r_conciseness` and the generation budget bound the other end.
    """
    n = len(reasoning_text(completion))
    if n <= 0:
        return 0.0
    span = max(w.reason_target_chars - w.reason_min_chars, 1)
    return max(0.0, min(1.0, (n - w.reason_min_chars) / span))


def _name_tokens(name: str) -> set[str]:
    """Distinctive words of a GO name — the ones whose presence means something was explained."""
    stop = {"activity", "process", "protein", "binding", "cellular", "regulation", "positive",
            "negative", "involved", "response", "complex", "component", "molecular", "function"}
    return {t for t in re.findall(r"[a-z]{5,}", name.lower()) if t not in stop}


def r_reasoning_faithfulness(completion: str, go_names: dict[str, str] | None) -> float:
    """Share of the ANSWER's GO ids that the reasoning actually accounts for. Bounded [0,1].

    A term counts as accounted for when the trace names its id, or uses a distinctive word from its
    GO name. This is the term that makes a trace worth showing to a reviewer: it separates a model
    that reasons and then answers from one that reasons about nothing and answers anyway. It is
    deliberately checkable after the fact rather than a judgement call.

    Returns 0.0 when the answer names no terms, so an empty answer cannot score full marks by
    vacuous quantification.
    """
    answer = extract_go_terms(completion, final_answer_only=True)
    if not answer:
        return 0.0
    trace = reasoning_text(completion).lower()
    if not trace:
        return 0.0
    trace_words = set(re.findall(r"[a-z]{5,}", trace))
    accounted = 0
    for term in answer:
        if term.lower() in trace:
            accounted += 1
            continue
        tokens = _name_tokens((go_names or {}).get(term, ""))
        if tokens and tokens & trace_words:
            accounted += 1
    return accounted / len(answer)


def r_truncated(completion: str) -> float:
    """1.0 when the budget cut the answer mid-identifier, else 0.0. A penalty, not a reward.

    466 of 472 stored rollouts ended partway through a GO id (`GO:01`), which means the answer was
    never finished and the trailing fragment is unscorable. Without this term, raising the budget to
    make room for reasoning just moves the cut rather than removing it.
    """
    return 1.0 if re.search(r"GO:\d{0,6}$", completion.rstrip()) else 0.0


def propagate_ancestors(go_terms: set[str], obo_ancestors: dict[str, set[str]]) -> set[str]:
    """Expand each GO term with its DAG ancestors (True-Path Rule)."""
    out = set(go_terms)
    for t in go_terms:
        out |= obo_ancestors.get(t, set())
    return out


def strip_implied_ancestors(go_terms: set[str], obo_ancestors: dict[str, set[str]]) -> set[str]:
    """Inverse of `propagate_ancestors`: drop terms already implied by a more specific sibling.

    Scoring propagates predictions (cafaeval's `pred_parser` runs `propagate` with `prop_mode`), so
    naming an ancestor explicitly adds nothing that propagation would not add for free. The set
    returned here therefore scores identically to `go_terms` while being much smaller — which
    matters when a fixed generation budget is what limits how many terms a model can emit.
    """
    implied: set[str] = set()
    for term in go_terms:
        implied |= obo_ancestors.get(term, set()) - {term}
    return set(go_terms) - implied


def ancestor_redundancy(completion: str, obo_ancestors: dict[str, set[str]],
                        final_answer_only: bool = False) -> float:
    """Fraction of emitted GO ids that another emitted id already implies (0.0 = all specific).

    Scoring propagates predictions, so an explicitly named ancestor earns nothing that propagation
    would not have added for free — measured on the Phase 4 rollouts, deleting them moved weighted
    F_max by exactly 0.000000. They are not free in *generation budget* though: at the operating point
    of 64 tokens the model emits about 5 ids, so every redundant one displaces a leaf it could have
    named instead. This makes that displacement visible to the optimiser.
    """
    emitted = extract_go_terms(completion, final_answer_only)
    if not emitted:
        return 0.0
    specific = strip_implied_ancestors(emitted, obo_ancestors)
    return (len(emitted) - len(specific)) / len(emitted)


def ia_weighted_f1(pred: set[str], true: set[str], ia: dict[str, float] | None) -> float:
    """IA-weighted F1 over (already ancestor-propagated) GO-term sets.

    ia=None → plain F1; with ia, each term is weighted by its information accretion, matching
    cafaeval's weighted precision/recall (the eval-time metric this proxies).
    """
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0

    def w(terms):
        return sum(ia.get(t, 1.0) for t in terms) if ia else float(len(terms))

    tp = pred & true
    wp, wt = w(pred), w(true)   # guard zero WEIGHT, not just emptiness: IA weights can be 0
    prec = w(tp) / wp if wp > 0 else 0.0   #   (e.g. GO roots have IA 0.0 in IA.txt)
    rec = w(tp) / wt if wt > 0 else 0.0
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# Reward variants, selected by SENPAI_REWARD_VARIANT.
#
# "union" (default) is the shipped reward: one IA-weighted F1 over the union of go_mf/go_bp/go_cc.
# It is aspect-blind, while the evaluation metric is the MEAN of per-aspect F_max. That mismatch is
# not cosmetic — it lets the policy raise the union F1 by piling into whichever aspect is cheapest.
# CC has 4,043 terms against MF's 11,263 and BP's 27,942, and GRPO was observed converting MF into
# CC: at 100 steps CC nearly doubled while MF fell 27%, leaving the aspect-mean metric flat.
#
# "aspect_mean" scores each aspect separately and averages, matching how the metric aggregates, so
# trading MF away for CC no longer looks free to the optimiser.
REWARD_VARIANTS = ("union", "aspect_mean", "aspect_mean_specific", "aspect_mean_reasoned")
DEFAULT_REWARD_VARIANT = "union"

# Variants that score the reasoning trace as well as the answer. Kept as data rather than an `if`
# so train.py can ask whether it must load GO names before building the reward. The weights each
# variant implies stay in train.py with the rest of the wiring, so there is one source of truth.
REASONED_VARIANTS = ("aspect_mean_reasoned",)
ASPECT_AWARE_VARIANTS = ("aspect_mean", "aspect_mean_specific", "aspect_mean_reasoned")


def active_reward_variant() -> str:
    """Read SENPAI_REWARD_VARIANT, failing closed on an unknown name."""
    import os

    name = os.environ.get("SENPAI_REWARD_VARIANT", DEFAULT_REWARD_VARIANT).strip()
    if name not in REWARD_VARIANTS:
        raise ValueError(
            f"SENPAI_REWARD_VARIANT={name!r} is not a known reward variant; "
            f"expected one of {list(REWARD_VARIANTS)}"
        )
    return name


def r_fmax_aspect_mean(completion: str, true_go: set[str],
                       obo_ancestors: dict[str, set[str]], ia: dict[str, float] | None,
                       go_aspects: dict[str, str], final_answer_only: bool = False) -> float:
    """Mean per-aspect IA-weighted F1 over propagated sets — mirrors the metric's aggregation.

    Aspects with no ground-truth term for this protein are skipped rather than scored 0, exactly as
    `aggregate_weighted_fmax` averages only the aspects present.
    """
    pred = propagate_ancestors(extract_go_terms(completion, final_answer_only), obo_ancestors)
    true = propagate_ancestors(true_go, obo_ancestors)
    scores = []
    for aspect in ("MF", "BP", "CC"):
        aspect_true = {t for t in true if go_aspects.get(t) == aspect}
        if not aspect_true:
            continue
        aspect_pred = {t for t in pred if go_aspects.get(t) == aspect}
        scores.append(ia_weighted_f1(aspect_pred, aspect_true, ia))
    return sum(scores) / len(scores) if scores else 0.0


def r_fmax(completion: str, true_go: set[str], obo_ancestors: dict[str, set[str]],
           ia: dict[str, float] | None, final_answer_only: bool = False) -> float:
    """Per-rollout IA-weighted-F1 proxy for cafaeval F_max. Calibrate correlation in Stage 0."""
    pred = propagate_ancestors(extract_go_terms(completion, final_answer_only), obo_ancestors)
    true = propagate_ancestors(true_go, obo_ancestors)
    return ia_weighted_f1(pred, true, ia)


def make_reward_fn(obo_ancestors: dict[str, set[str]], ia: dict[str, float] | None,
                   weights: RewardWeights | None = None, final_answer_only: bool = False):
    """Return a TRL-style reward_fn(prompts, completions, **cols) -> list[float].

    `cols` carries the per-sample dataset columns (incl. ground-truth go_bp/go_mf/go_cc), passed
    through by GRPOTrainer. Ground-truth labels are used ONLY on RL-train prompts (see data.py
    split contract). Build obo_ancestors/ia once via go_obo.load_go_ancestors / load_ia_weights.
    """
    w = weights or RewardWeights()

    def reward_fn(prompts, completions, **cols):
        gts = _ground_truth_go(cols)
        out = []
        for i, comp in enumerate(completions):
            true_go = gts[i] if i < len(gts) else set()
            out.append(
                reward_components(
                    comp, true_go, obo_ancestors, ia, w, final_answer_only
                ).total
            )
        return out

    return reward_fn


def reward_components(
    completion: str,
    true_go: set[str],
    obo_ancestors: dict[str, set[str]],
    ia: dict[str, float] | None,
    weights: RewardWeights | None = None,
    final_answer_only: bool = False,
    go_aspects: dict[str, str] | None = None,
    go_names: dict[str, str] | None = None,
) -> RewardComponents:
    """Return the exact composite and its independently loggable components.

    Passing `go_aspects` selects the aspect-mean fmax term; omitting it keeps the shipped union
    reward, so the default behaviour is unchanged. `go_names` only affects the faithfulness term,
    which is weighted 0 unless the caller asked for a reasoned variant.
    """
    w = weights or RewardWeights()
    if go_aspects is not None:
        fmax = r_fmax_aspect_mean(
            completion, true_go, obo_ancestors, ia, go_aspects, final_answer_only
        )
    else:
        fmax = r_fmax(completion, true_go, obo_ancestors, ia, final_answer_only)
    format_score = r_format(completion)
    conciseness = r_conciseness(completion, w)
    # ALWAYS measured, even at lambda_spec=0 where it does not affect the total. Computing it
    # conditionally made unpenalised runs log a constant 0.0 that was indistinguishable from a real
    # measurement of zero, so the penalised arm had no baseline to be compared against. The same
    # argument applies to the three reasoning components below, so they are also always measured.
    redundancy = ancestor_redundancy(completion, obo_ancestors, final_answer_only)
    substance = r_reasoning_substance(completion, w)
    faithfulness = r_reasoning_faithfulness(completion, go_names)
    truncation = r_truncated(completion)
    # Substance and faithfulness multiply rather than add: a long trace that explains none of the
    # answer and a faithful one-liner are both worthless, and summing would pay for either alone.
    total = (fmax + w.lambda_fmt * format_score + w.lambda_len * conciseness
             - w.lambda_spec * redundancy
             + w.lambda_reason * substance * faithfulness
             - w.lambda_truncation * truncation)
    return RewardComponents(total, fmax, format_score, conciseness, redundancy,
                            substance, faithfulness, truncation)


def validate_reward_group(values: list[float], *, require_non_degenerate: bool = True) -> None:
    """Fail a smoke contract when grouped rewards are non-finite or all identical."""
    if not values or not all(isfinite(value) for value in values):
        raise ValueError(f"reward group must be non-empty and finite: {values}")
    if require_non_degenerate and len(values) > 1 and len(set(values)) == 1:
        raise ValueError(f"reward group is degenerate: {values}")


def _coerce_go_list(value) -> set[str]:
    """A go_bp/go_mf/go_cc cell may be a list, a stringified list, or plain text."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {t for t in value if isinstance(t, str) and GO_RE.fullmatch(t)}
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return _coerce_go_list(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                pass
        return set(GO_RE.findall(s))
    return set()


def _ground_truth_go(cols) -> list[set[str]]:
    """Merge go_mf/go_bp/go_cc columns into a per-sample ground-truth GO set.

    Mirrors bioreason2 evals/cafa_evals.extract_reasoning_ground_truth (reasoning mode).
    """
    present = [c for c in _GT_COLUMNS if c in cols]
    if not present:
        return []
    n = len(cols[present[0]])
    merged = []
    for i in range(n):
        s: set[str] = set()
        for c in present:
            s |= _coerce_go_list(cols[c][i])
        merged.append(s)
    return merged
