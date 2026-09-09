# Vendored: GO-GPT (from `bowang-lab/BioReason-Pro`)

plan.md Phase 5 (ADR-028): GO-GPT is the paper's own open-sourced GO-term-prediction model,
used here as a zero-LLM baseline to check what BioReason-Pro's reasoning layer adds over a
much cheaper discrete annotator.

- **Source**: https://github.com/bowang-lab/BioReason-Pro
- **Pinned commit**: `a93ac2a96387103c100c4d6a6674d6fe1dfaf9e8` (`main`, fetched 2026-08-23)
- **Vendored path upstream**: `gogpt/src/gogpt/` (verbatim, byte-for-byte — fetched via
  `gh api repos/bowang-lab/BioReason-Pro/contents/...`, not a summarizing tool, specifically so
  the code run here is provably identical to what's on GitHub)
- **License (code)**: MIT — see `LICENSE` in this directory (copied verbatim from the upstream
  repo root; `gogpt/` carries no separate license file of its own, and is an ordinary tracked
  subdirectory of the MIT-licensed parent repo, not a git submodule — confirmed via
  `gh api repos/bowang-lab/BioReason-Pro` returning no `.gitmodules`).
- **License (weights)**: Apache-2.0 — `wanglab/gogpt` on HuggingFace, `gated: false` (confirmed
  via the HF REST API `https://huggingface.co/api/models/wanglab/gogpt`). Its own frozen encoder
  dependency, `facebook/esm2_t36_3B_UR50D`, is MIT-licensed and also `gated: false`.

## Why vendored instead of reimplemented

Every other upstream reference used in this project (InterPro fetching, prompt templates) was
read for understanding and reimplemented fresh, calling the same public data sources rather than
copying code — see plan.md ADR-028's own framing for `interpro_api.py`. GO-GPT is treated
differently on purpose: its architecture (`PrefixCausalAttention`'s dual protein/GO streams,
per-aspect start/end tokens, gated attention, beam-search decoding tied to a specific checkpoint's
trained weights) is custom, non-trivial model code, not a REST client. Reimplementing it from a
written description risks a subtle divergence (wrong layer index, wrong masking, wrong tokenizer
vocabulary) that would silently produce plausible-looking but wrong GO-term predictions —
invalidating the baseline without any obvious signal that something was off. Running the actual
tested code against the actual checkpoint it was trained with is the only way to be confident the
baseline measures GO-GPT, not this project's guess at GO-GPT. The code's own MIT license was
written for exactly this kind of reuse.

## What's vendored, and why each file is needed

`gogpt/src/gogpt/__init__.py` eagerly imports every name below, so all of them must be present
for `import gogpt` to succeed even though only `inference.py` and `models/gogpt.py` are on the
call path this project actually uses (`GOGPTPredictor.from_pretrained(...).predict(...)`):

- `inference.py` — `GOGPTPredictor`, the high-level API this project calls.
- `models/gogpt.py` — the `GOGPT` model itself (used).
- `models/gogpt_lightning.py` — `LightningGOGPT`, upstream's training wrapper (imported, never
  instantiated here — this project only runs inference).
- `config/model_config.py` — `GOGPTConfig` dataclass (used, indirectly, by `inference.py`).
- `data/dataset.py`, `data/tokenizer.py`, `data/preprocessing_utils.py` — upstream's dataset-
  building utilities (imported, never called here).
- `utils/organism_mapper.py` — `OrganismMapper` (imported; the actual `from_pretrained` path uses
  a separate lightweight `OrganismMapperJSON` defined inline in `inference.py` instead).

None of these files were modified. `gogpt/src/gogpt/{data,models,utils,config}/` ship with no
`__init__.py` upstream either (Python namespace packages) — reproduced exactly, not an omission.

## How this project calls it

`scripts/run_gogpt_baseline.py` puts `third_party/gogpt/src` on `sys.path` (the same mechanism
upstream's own `gogpt_api.py` uses for `gogpt/src`) and calls
`gogpt.inference.GOGPTPredictor.from_pretrained("wanglab/gogpt")` directly — not upstream's
`gogpt_api.py` wrapper, which additionally imports `bioreason2.dataset.cafa5.processor` (a sibling
package in the upstream monorepo, not vendored here) purely to pretty-print GO term names. This
project only needs raw `GO:#######` ids for scoring, so that dependency is skipped entirely.

`GOGPTPredictor.from_pretrained` and `GOGPT.__init__` (both unmodified) call
`AutoTokenizer`/`AutoModel.from_pretrained` on the encoder path (`config.yaml`'s
`embed_model_path`, `facebook/esm2_t36_3B_UR50D`) with **no `revision` argument at all** — there is
no hook to force `approved_assets.json`'s pinned revision for that specific model without editing
vendored code. `scripts/run_gogpt_baseline.py` still checks the encoder **name** is approved before
loading (the property that actually matters: never load an arbitrary, unapproved encoder), and
accepts HF's own `main` resolution for the rest, same as upstream's own code does. GO-GPT's own
weights (`wanglab/gogpt` itself) don't have this gap — `from_pretrained(model_id, revision=...)`
does take and use a revision there.

## To update the pin

Re-fetch `gogpt/src/gogpt/` from a newer upstream commit, diff file-by-file against what's here,
and update the commit SHA above. Do not hand-edit the vendored files.
