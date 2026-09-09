# The BioReason-Pro paper — reference record

**This is the paper this repository is a license-contained reimplementation of.**

- **Preprint:** https://www.biorxiv.org/content/10.64898/2026.03.19.712954v1.full (PDF re-read
  2026-08-21; the HTML `.full` view truncates before the methods/availability sections, so treat any
  fact sourced only from that view as unverified)
- **Upstream code:** https://github.com/bowang-lab/BioReason-Pro
- **Licence:** the preprint carries a standard bioRxiv CC-BY 4.0 International notice, present in the
  page footer of the PDF (linking to `creativecommons.org/licenses/by/4.0/`) and repeated throughout.
  **This corrects the previous record on this page**, which reported no licence statement — that
  finding came from the HTML view, which does not surface the footer. CC-BY permits reuse with
  attribution; this document still paraphrases rather than reproduces the text, consistent with this
  project's practice for every source regardless of its licence.

## The paper's setup, as reported

| Dimension | The paper |
|---|---|
| **Test split** | Temporal, CAFA-style (Zhou et al., 2019). Trained on annotations through **November 2022**. The paper defines two knowledge tiers and evaluates only the stricter one: **no-knowledge** proteins (no prior experimental annotation in *any* GO aspect) that gained annotation in the target aspect between **March 2023 and February 2024**. *Limited-knowledge* proteins (annotated in other aspects already) are explicitly excluded from evaluation, because the reasoning-trace corpus is built from cross-aspect annotations and could leak into the held-out aspect. Final set: **8,630 proteins**, 230,824 annotations, mean 26.75 / median 18.0 GO terms per protein. |
| Protein encoder | ESM3-1B, frozen — chosen over ESM2 specifically for structure-awareness (it also consumes PDB/predicted structure; ~10% of training proteins had none, and dropping structure at inference only mildly hurt accuracy). |
| LLM backbone | Qwen3-4B-Thinking. |
| **Prompt inputs** (§4.3.5) | ESM3 residue embeddings, 200 GO-graph node embeddings, **organism** (text), **InterPro** domains with residue ranges (via InterProScan), **GO-GPT's own greedy GO predictions** (text), and optional **STRING** PPI partners. Structure coordinates feed ESM3's encoder, not the text prompt. |
| **Model output** | a reasoning trace, then a structured final answer: functional summary, GO terms across MF/BP/CC, and hypothesised interaction partners. |
| Reasoning supervision | GPT-5-generated traces over 130K+ proteins for SFT. |
| Training data | 133,492 proteins, 3,135 organisms, from UniProt + GOA (Nov 2022 release), restricted to experimental/curated evidence codes per CAFA convention. A SwissProt-only ablation was tried for both GO-GPT and the main corpus and did not help. |
| **RL algorithm** | Built on GRPO, adopting GSPO's sequence-level (not token-level) importance-sampling correction and Dr.GRPO's length-bias correction and DAPO's Clip-Higher exploration bound; the supplementary hyperparameter table names the resulting recipe **DR-GRPO**. Advantages are group-centered but normalised by the **global batch** reward std-dev rather than per group, stated as a deliberate fix for within-group variance collapsing to near zero under plain grouping. Group size 24, temperature 1.0. |
| **RL reward** | a single scalar: weighted F_max between GO terms regex-extracted from the model's structured final-answer block — **explicitly excluding the reasoning trace** — and ground truth. No format, length, or reasoning-quality term is described anywhere in the methods or supplement. |
| **GO-GPT** | the paper's own baseline model, not a third-party tool: a frozen ESM2-3B encoder feeding a 12-layer autoregressive GO-term decoder, one model spanning all three aspects, conditioned on a learned per-organism embedding. Its own greedy predictions are one of the fields injected into BioReason-Pro's prompt. |
| **Reported GO metric** (weighted F_max, this same 8,630-protein set) | InterLabelGO+ 0.63 · GO-GPT 0.65 greedy → 0.70 best-of-10 (oracle-selected) · BioReason-Pro SFT 0.64 greedy → 0.67 oracle · BioReason-Pro RL 0.66 greedy → 0.67 oracle. Unweighted F_max is reported separately and reaches up to ~0.74–0.76 depending on system — this is almost certainly the true source of the abstract's standalone "73.6%" figure, which the body never labels as weighted or unweighted and which does not match any entry in the paper's own weighted table. |
| **LLM-judge score** | GPT-5.1, deterministic, schema-constrained, 5 axes (MF/BP/CC correctness, specificity, reliability): RL 8.03/10, SFT 7.65/10, a non-reasoning baseline (Prot2Text-v2) 4.15/10 — RL beats SFT on every axis, paired by protein (n≈8,159). |
| **Human-expert evaluation** | 27 blinded molecular-biology reviewers, 162 test proteins. Tie-or-exceed rate against UniProt annotations: SFT 79%, RL 73% (not a significant difference). On a separate 7-axis 1–10 rubric (correctness, reasoning/evidence attribution, hallucination avoidance, mechanistic depth, PPI plausibility): **SFT averages 8.0, RL 7.4 — SFT scores higher than RL on the paper's own human rubric**, the opposite ranking from the automated judge. |

## Whose data and code this is

The paper's own Data/Code/Model-Availability statement routes everything through custom
`bioreason.net/{data,code,gogpt,sft,rl,atlas}` redirect links. **It never once writes "wanglab",
"bowang-lab", or a literal `huggingface.co`/`github.com/bowang-lab` URL** in its text or in any of its
89 embedded hyperlinks — so the paper's own text does not confirm that this repository's data sources
are its release.

The identification rests on independent, mutually-corroborating circumstantial evidence instead:
the `wanglab` Hugging Face org page is titled "WangLab UofT" and links directly to
`github.com/bowang-lab`; author **Bo Wang** is on the paper; the `wanglab/bioreason-pro-sft-reasoning-data`
and `wanglab/bioreason-pro-test-data` dataset cards themselves cite this exact preprint and link to
`github.com/bowang-lab/BioReason-Pro`; the sealed test set's protein count (8,630) matches the paper's
stated final holdout size exactly; and the upstream GitHub README states outright that these HF
datasets are its own released artifacts. No step in this chain is a live check of where the paper's
own `bioreason.net` links actually redirect — that remains open. Treat the identification as
**plausible and well-corroborated, not paper-confirmed.**

## What that means for this repository

**The split was never the defect.** `wanglab/bioreason-pro-test-data` is, or very closely approximates,
the paper's own temporal no-knowledge holdout (see `plan.md` ADR-024). What was wrong was this
project's *validation* split, a random partition of an over-annotated corpus.

**The comparator was wrong.** This repository's sealed weighted F_max (`0.32674`) should be read
against the paper's own weighted-F_max range (**0.63–0.70**, not 73.6%, which is very likely an
unweighted figure). The size of the remaining gap is smaller than previously stated, and now has a
cheap external floor to check against: GO-GPT alone, a discrete annotator with no LLM and no
reasoning, scores within that same range — the paper's own results say an LLM reasoning layer barely
clears a much cheaper baseline, which is a useful caution for this project's own conclusions.

**Two output dimensions are absent here entirely:** the functional summary with its LLM-judge score,
and the human-preference study. This repository scores GO terms only.

**Three prompt inputs are absent here:** organism, PDB structure, and GO-GPT hypotheses. Upstream's
own code makes two of these, plus InterPro and STRING, far cheaper to add than previously assessed —
`interpro_api.py` calls the free public EBI InterProScan REST service with no licensing restriction
found in the code, and GO-GPT is upstream's own fully open-sourced model, self-hostable on a single
GPU via HF weights with no gating. Organism is a plain text field with no dependency at all.

**The RL recipe differs from what this project assumed on two counts.** The paper's RL reward is a
single F_max term scored only against the final answer, never the reasoning trace — this repository's
`rewards.py` format/length/reasoning-quality terms have no paper counterpart and are local additions,
one of which (the format term) this project's own rollout analysis found to be measurably inert. And
the paper normalises GRPO advantages across the whole batch rather than per rollout group, a specific
variant this project has not tried, distinct from the token-vs-sequence importance-ratio question this
project's training loop structurally cannot exercise (ADR-008).

**The paper's own RL step did not uniformly improve on SFT.** RL beat SFT on the automated judge but
lost to it on the human-rubric evaluation. This repository has neither harness yet, but should not
assume RL strictly dominates SFT once either exists.

See [`../../plan.md`](../../plan.md) ADR-022 through ADR-028 for the full gap analysis and what is
being done about it.
