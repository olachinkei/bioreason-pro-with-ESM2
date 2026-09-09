"""train.py — PRIMARY EDITABLE entrypoint (SFT + RL).

Wraps the public SFT loop (bioreason2 train_protein_llm.py, PyTorch Lightning) and adds the
authored RL/GRPO stage. Changes to the training loop belong here; bioreason_pro/, eval.py,
data.py, data/ are Protected.

SKELETON: RunArgs, SENPAI env ceilings, stage dispatch, GRPO/GSPO config assembly, W&B init,
timeout-safe checkpoint+eval, and the exact SENPAI-RESULT marker are wired; loop bodies are TODO.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass

import simple_parsing


@dataclass
class RunArgs:
    stage: str = "sft"                    # {sft, rl}
    rl_algo: str = "grpo"                 # {grpo (baseline, paper-faithful), gspo (opt-in variant)}
    use_multimodal: bool = True           # false = text-only (Stage 0/1)
    seed: int = 0                         # PARTITIONS THE DATA (protein_split hash) — freeze it
    # Stochastic training only: LoRA init, dropout, rollout sampling. -1 = follow `seed`, which
    # is the shipped behaviour. Kept separate because `seed` also decides which proteins land in
    # val, so varying it would repartition the eval set and make the number incomparable with
    # every measurement in plan.md instead of measuring training variance.
    train_seed: int = -1
    # ADR-015. False reproduces every SFT in plan.md exactly. True drops the manufactured empty
    # `<think></think>` from the loss so the backbone's own reasoning survives supervision, which is
    # the precondition for the reasoning reward (ADR-014) having anything to shape.
    mask_empty_think: bool = False
    epochs: int = 10
    data_subset_frac: float = 1.0
    resume_from: str = ""                 # W&B artifact ref to resume optimizer/step state
    model_name: str = ""                  # override backbone (e.g. Qwen/Qwen3-0.6B for Stage-0 smoke)
    # RL / GRPO knobs
    use_vllm: bool = True                 # False for the CPU/HF-generation smoke (colocate vLLM for scale)
    smoke: bool = False                   # use the tiny built-in RL dataset (no HF gated access)
    num_generations: int = 8
    max_completion_length: int = 256
    learning_rate: float = 1e-5
    rl_temperature: float = 1.0           # multimodal GRPO rollout sampling temperature
    gradient_accumulation_steps: int = 4  # SFT: micro-batch=1 protein/step, accumulate to this
    streaming_steps_per_epoch: int = 500  # finite logical epoch for custom streaming loops
    max_seq_len: int = 10000              # SFT: skip examples whose tokenized length exceeds this
    gradient_checkpointing: bool = True   # trade compute for memory (needed for the 4B backbone)
    max_steps: int = -1                   # >0 overrides epochs (smoke)
    rl_num_prompts: int = 64
    output_dir: str = ""                  # explicit artifact path; defaults under outputs/
    # W&B (rank-0 only)
    wandb_name: str = ""
    wandb_group: str = ""
    agent: str = ""
    # ceilings are ALSO read from env (SENPAI_*); the min() wins.
    eval_subset_size: int = 256
    # Multimodal assembly is restricted to the allowlisted MIT-licensed ESM2-650M.
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm_layer: int = 33
    freeze_esm: bool = True
    sft_artifact: str = ""                # RL: immutable entity/project/SFT-artifact:vN input
    artifact_root: str = ""               # shared cache for use_artifact downloads
    # Populated internally after the active RL run calls use_artifact. Direct CLI paths are rejected.
    resolved_sft_artifact: str = ""
    sft_adapter: str = ""
    # LoRA (paper: SFT r=128/α=256, RL r=16/α=32; dropout 0.05) + RL KL beta (paper default 0.0).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    beta: float = 0.0
    # ADR-036/037: the paper batches B=8 proteins x G=24 rollouts/step (192 total), pooling advantage
    # normalization across the whole batch — this project's loop drew exactly 1 protein/step with no
    # cross-protein pooling at all. proteins_per_step=1 (default) reproduces today's exact behaviour.
    proteins_per_step: int = 1
    # "group_mean": today's Dr.GRPO baseline (no std division). "group_mean_global_std": paper's own
    # Eq. 4.19-4.20 — same per-protein group-mean numerator, divided by ONE std pooled across the
    # whole proteins_per_step*num_generations batch. Independent of proteins_per_step so either can be
    # changed alone (a batched-but-still-Dr.GRPO run, or a std-normalized single-protein run, are both
    # valid, individually attributable configurations).
    rl_advantage_normalization: str = "group_mean"

    def resolved_epochs(self) -> int:
        env = os.environ.get("SENPAI_MAX_EPOCHS")
        if not env:
            return self.epochs
        ceiling = int(env)
        if ceiling <= 0:
            raise ValueError("SENPAI_MAX_EPOCHS must be a positive integer")
        return min(self.epochs, ceiling)

    def resolved_max_steps(self) -> int:
        """Return a finite optimizer-step bound for the custom streaming SFT/RL loops.

        Hugging Face iterable datasets do not expose a stable post-filter length, so a logical
        epoch is an explicit number of optimizer updates. `--max_steps` remains the exact
        override used by smoke tests; otherwise epochs, the subset fraction, and the Senpai epoch
        ceiling all contribute to a deterministic finite bound.
        """
        if self.max_steps > 0:
            return self.max_steps
        steps_per_epoch = max(
            1, math.ceil(self.streaming_steps_per_epoch * self.data_subset_frac)
        )
        return self.resolved_epochs() * steps_per_epoch

    def timeout_deadline(self) -> float | None:
        mins = os.environ.get("SENPAI_TIMEOUT_MINUTES")
        return (time.monotonic() + float(mins) * 60) if mins else None

    def resolved_train_seed(self) -> int:
        """Seed for stochastic training. Defaults to `seed`, so shipped runs are unchanged."""
        return self.seed if self.train_seed < 0 else self.train_seed

    def validate(self, *, require_training_dataset: bool = True) -> None:
        assert self.stage in ("sft", "rl"), self.stage
        assert self.rl_algo in ("grpo", "gspo"), self.rl_algo
        assert 0 < self.data_subset_frac <= 1.0
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.streaming_steps_per_epoch <= 0:
            raise ValueError("streaming_steps_per_epoch must be positive")
        if self.max_steps != -1 and self.max_steps <= 0:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.proteins_per_step < 1:
            raise ValueError("proteins_per_step must be >= 1")
        if self.rl_advantage_normalization not in ADV_NORMALIZATION_MODES:
            raise ValueError(
                f"--rl_advantage_normalization {self.rl_advantage_normalization!r} must be one of "
                f"{list(ADV_NORMALIZATION_MODES)}"
            )
        from bioreason_pro.license_policy import (
            require_immutable_wandb_artifact_ref,
            require_approved_dataset,
            require_approved_model,
            validate_checkpoint_layout,
        )
        from bioreason_pro.model import ModelConfig

        require_approved_model(self.model_name or ModelConfig.text_model_name, "text")
        if self.use_multimodal:
            require_approved_model(self.esm_model_name, "protein")
        if require_training_dataset and not self.smoke:
            repo = SFT_DATA_REPO if self.stage == "sft" else RL_DATA_REPO
            require_approved_dataset(repo, f"{self.stage}-training")
        if self.stage == "rl" and self.use_multimodal:
            if not self.sft_artifact:
                raise ValueError(
                    "multimodal RL requires --sft_artifact "
                    "entity/project/artifact:vN; local --sft_adapter hand-offs are unsupported"
                )
            require_immutable_wandb_artifact_ref(self.sft_artifact, "SFT input artifact")
            if self.sft_adapter and not self.resolved_sft_artifact:
                raise ValueError(
                    "--sft_adapter is internal; the active W&B run must resolve --sft_artifact "
                    "with use_artifact"
                )
            if self.resolved_sft_artifact:
                require_immutable_wandb_artifact_ref(
                    self.resolved_sft_artifact, "resolved SFT input artifact"
                )
                if self.resolved_sft_artifact != self.sft_artifact:
                    raise ValueError("resolved SFT artifact does not match the requested version")
        if self.sft_adapter:
            validate_checkpoint_layout(
                self.sft_adapter,
                expected_text_model=self.model_name or ModelConfig.text_model_name,
                expected_protein_model=self.esm_model_name,
            )


def build_grpo_config(args: RunArgs):
    """Assemble GRPOConfig. Baseline = GRPO (token-level IS); gspo flips to sequence-level.

    NOTE: pin the TRL version exposing importance_sampling_level/loss_type. On current TRL weight
    sync is VLLMGeneration.sync_weights(); colocate sleep mode is opt-in via vllm_enable_sleep_mode.
    """
    from trl import GRPOConfig  # pinned dep

    common = dict(
        output_dir=args.output_dir or f"outputs/rl-{args.wandb_name or 'run'}",
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,  # must be a multiple of num_generations
        beta=args.beta,                   # KL off by default (paper); --beta to pin
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",                # W&B Models curves (reward, kl, per-component)
        seed=args.resolved_train_seed(),
    )
    if args.max_steps > 0:
        common["max_steps"] = args.max_steps
    else:
        common["num_train_epochs"] = args.resolved_epochs()
    if args.use_vllm:                     # colocate vLLM for scale; smoke uses HF generation
        common.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3,
                      vllm_enable_sleep_mode=True)
    if args.rl_algo == "gspo":
        # GSPO = sequence-level IS; MUST also set loss_type="grpo" (TRL issue #3823) + eps.
        common.update(importance_sampling_level="sequence", loss_type="grpo",
                      epsilon=3e-4, epsilon_high=4e-4)
    else:
        common.update(importance_sampling_level="token")  # standard GRPO (paper-faithful)
    return GRPOConfig(**common)


def _sft_smoke_dataset() -> list[dict]:
    """Checked-in multimodal SFT rows that use the same sequence/GO-only contract."""
    import data

    return data.synthetic_fixture_rows()


EMPTY_THINK_SPAN = "<think>\n\n</think>"


def _mask_empty_think_labels(ids: list[int], labels: list[int], tok) -> tuple[list[int], int]:
    """Drop the manufactured empty `<think></think>` from the SFT loss. Returns (labels, n_masked).

    ADR-015. `format_go_answer` writes no reasoning, and the chat template's final-assistant branch
    renders any target without `</think>` as `<think>\\n\\n</think>\\n\\n` + the answer. So every
    supervised example was explicitly teaching the model to open and immediately close its reasoning
    block — and it learned it exactly: 472 of 472 stored rollouts, and 16 of 16 in the Phase 11
    smoke at budget 512, emit an empty trace. RL cannot undo that, because GRPO only reweights what
    the policy samples and the probability of a non-empty trace is ~0.

    Masking rather than deleting keeps the token sequence identical to what inference will see, so
    the SFT distribution is unchanged; it only stops the gradient that was reinforcing emptiness.
    Nothing is fabricated here — no reasoning text is invented, we merely stop teaching its absence.
    """
    span = tok.encode(EMPTY_THINK_SPAN, add_special_tokens=False)
    if not span:
        return labels, 0
    out = list(labels)
    masked = 0
    start = 0
    while True:
        i = _find_subsequence(ids, span, start)
        if i == -1:
            break
        for j in range(i, i + len(span)):
            if out[j] != -100:
                out[j] = -100
                masked += 1
        start = i + len(span)
    return out, masked


def _find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def _sft_forward_inputs(example: dict, tok, template: str, dev: str, multimodal: bool, max_seq_len: int = 10000,
                        mask_empty_think: bool = False):
    """Build one supervised example (micro-batch of 1): render the full chat (assistant turn INCLUDED,
    add_generation_prompt=False), expand protein pads (multimodal), tokenize, and mask labels to the
    assistant span (reasoning+answer). Returns model.forward kwargs, or None if there's nothing to learn
    or the example exceeds `max_seq_len` (skipped to bound memory/time)."""
    import torch

    import data

    formatted = example if isinstance(example.get("prompt"), list) else data.format_cafa5_for_protein_llm(example)
    if multimodal:
        text = tok.apply_chat_template(formatted["prompt"], tokenize=False,
                                       add_generation_prompt=False, chat_template=template)
        seq = formatted["protein_sequences"][0]
        text = data.expand_pad_tokens(text, seq)
    else:
        user_text = next(c["text"] for c in formatted["prompt"][0]["content"] if c.get("type") == "text")
        chat = [{"role": "user", "content": user_text}, {"role": "assistant", "content": formatted["answer"]}]
        text, seq = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False), None
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"]
    if ids.shape[1] > max_seq_len:
        return None
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or -100)
    labels = data.mask_assistant_labels(
        ids[0].tolist(), tok.encode("<|im_start|>assistant\n", add_special_tokens=False),
        tok.encode("<|im_end|>", add_special_tokens=False), pad_id)
    n_think_masked = 0
    if mask_empty_think:
        labels, n_think_masked = _mask_empty_think_labels(ids[0].tolist(), labels, tok)
        # Fail loud rather than train a no-op arm: if the span never matches (a retokenisation, a
        # template change), the knob silently degrades to the shipped supervision and the run looks
        # like a completed experiment. Method rule 6.
        if n_think_masked == 0:
            raise ValueError(
                "mask_empty_think=True but the empty <think></think> span was not found in this "
                "example's tokens — the knob would have no effect and the run would silently be a "
                "duplicate of the shipped supervision. Check the chat template and tokenizer."
            )
    if all(x == -100 for x in labels):
        return None
    kw = dict(input_ids=ids.to(dev), attention_mask=enc["attention_mask"].to(dev),
              labels=torch.tensor([labels]).to(dev))
    if multimodal:
        kw.update(protein_sequences=[seq], batch_idx_map=[0])
    return kw


# Real datasets contain generated and aggregated columns in addition to sequence and GO labels.
# The approved contract deliberately drops every such column before prompt construction.
SFT_DATA_REPO = "wanglab/bioreason-pro-sft-reasoning-data"
RL_DATA_REPO = "wanglab/bioreason-pro-rl-reasoning-data"


def _adapt_real_row(r: dict, repo: str) -> dict:
    """Apply the reviewed sequence-only prompt and GO-label contract."""
    import data
    from bioreason_pro.data_contract import adapt_approved_training_row

    use = "sft-training" if repo == SFT_DATA_REPO else "rl-training"
    return adapt_approved_training_row(
        r, repo_id=repo, use=use, max_length_protein=data.MAX_LENGTH_PROTEIN
    )


def _stream_real_examples(args: RunArgs, repo: str, split: str, rank: int = 0, world: int = 1):
    """Yield adapted training/eval examples from a gated wanglab repo, filtered to `split` (leakage-safe
    via data.in_split) and SHARDED across `world` ranks (each rank sees a disjoint 1/world slice).
    Lazy — the caller bounds it by steps/timeout/subset."""
    import data
    from bioreason_pro.data_contract import MissingReasoningEvidence

    i = -1
    skipped = 0
    for r in data._load_stream(repo):
        if not r.get("protein_id") or not r.get("sequence"):
            continue
        if not data.in_split(r["protein_id"], split, args.seed):
            continue
        i += 1
        if world > 1 and i % world != rank:   # rank-sharded (round-robin over the filtered stream)
            continue
        try:
            yield _adapt_real_row(r, repo)
        except MissingReasoningEvidence:
            # A reasoned variant cannot supervise this row: no trace, or no evidence for its trace.
            # Skip and count. Job 971 died 12 minutes in on one such row, and the identical guard in
            # data.load_sft_dataset did not help because THIS is the path the SFT run actually uses
            # — the ADR-008 lesson (grep for the destination, not the declaration) applied to my own
            # fix. A run that skips everything is a broken variant, so the rate is reported below.
            skipped += 1
            if skipped in (1, 10, 100) or skipped % 1000 == 0:
                print(f"[data] skipped {skipped} row(s) lacking supervisable reasoning "
                      f"({skipped / max(i + 1, 1) * 100:.2f}% of {i + 1} seen)", flush=True)
            if i + 1 >= 200 and skipped / (i + 1) > 0.5:
                raise ValueError(
                    f"{skipped}/{i + 1} rows lack supervisable reasoning. That is a wrong variant "
                    "or a wrong dataset, not sparsity — refusing to train a decimated arm."
                ) from None


def _parse_go(v) -> set:
    """GO-term set from a list, a stringified list, or GO:ids in text (real cols are lists)."""
    from bioreason_pro.data_contract import parse_go_terms

    return parse_go_terms(v)


def _reasoning_diagnostics(records: list[dict]) -> dict:
    """Trace-quality metrics reported next to weighted_fmax. Never an input to training.

    `reasoning_nonempty` is the headline: Phases 7-9 shipped a recipe whose <think> block was empty
    in 472 of 472 stored rollouts, and no evaluation number said so. `reasoning_faithfulness` is the
    one a reviewer cares about — the share of predicted GO terms the trace actually accounts for.
    """
    from bioreason_pro import go_obo, rewards
    import eval as sealed_eval

    if not records:
        return {}
    try:
        names = go_obo.load_go_names(sealed_eval.OBO)
    except Exception:
        names = {}
    texts = [r.get("generated_response", "") for r in records]
    traces = [rewards.reasoning_text(t) for t in texts]
    nonempty = [t for t in traces if t]
    return {
        "reasoning_nonempty": len(nonempty) / len(texts),
        "reasoning_chars_mean": (sum(len(t) for t in nonempty) / len(nonempty)) if nonempty else 0.0,
        "reasoning_faithfulness": sum(
            rewards.r_reasoning_faithfulness(t, names) for t in texts
        ) / len(texts),
        "answer_truncated": sum(rewards.r_truncated(t) for t in texts) / len(texts),
    }


def _group_sd(values: list[float]) -> float:
    """Population sd of one rollout group — 0.0 means the component cannot teach anything.

    GRPO centres advantages within the group, so only a component that DISAGREES across the group
    reaches the gradient. See ADR-013: `r_format` scored an identical 1.0 in all 472 stored
    rollouts, so its 0.1 weight bought nothing for eight phases.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


ADV_NORMALIZATION_MODES = ("group_mean", "group_mean_global_std")
ADV_STD_EPS = 1e-6


def _pooled_advantages(group_rewards, normalization: str, eps: float = ADV_STD_EPS):
    """Per-protein group-mean-centred advantages, optionally pooled-std-normalized (ADR-036/037,
    paper Eq. 4.19-4.20). `group_rewards` is a list of 1-D reward tensors, one per protein in the
    step's batch (each of length `num_generations`).

    "group_mean" (default): reproduces the pre-ADR-037 Dr.GRPO baseline exactly — subtract each
    protein's own group mean, divide by 1.0 (no std normalization at all).
    "group_mean_global_std": same per-protein numerator, but divide by ONE population std pooled
    across every reward in the whole batch (all proteins' rollouts together), matching the paper's
    own formula. `global_std==0` (every reward in the batch is bit-identical) forces every per-protein
    numerator to 0 too, so this never produces a NaN — only a harmless 0/eps.

    Returns (list of per-protein advantage tensors, the pooled population std used for logging).
    """
    import torch

    all_r = torch.cat(list(group_rewards))
    global_std = all_r.std(correction=0)  # population std, matching _group_sd's convention
    if normalization == "group_mean_global_std":
        denom = global_std + eps
    elif normalization == "group_mean":
        denom = 1.0
    else:
        raise ValueError(f"unknown rl_advantage_normalization {normalization!r}")
    advantages = [(r - r.mean()) / denom for r in group_rewards]
    return advantages, global_std


def _append_prompt_suffix(user_turn: dict, suffix: str) -> dict:
    """Append variant guidance to the text item of a chat user turn, leaving other items alone."""
    if not suffix:
        return user_turn
    content = []
    appended = False
    for item in user_turn["content"]:
        if not appended and item.get("type") == "text":
            item = {**item, "text": f"{item['text']}\n\n{suffix.strip()}"}
            appended = True
        content.append(item)
    if not appended:
        raise ValueError("user turn has no text item to carry the prompt variant")
    return {**user_turn, "content": content}


def _eval_prompt_records(args: RunArgs, split: str, multimodal: bool) -> list[dict]:
    """Normalized eval prompts: {protein_id, user_turn (chat user message), sequence|None, gt_terms}.
    smoke → synthetic (no gated HF); else stream the SFT set filtered to `split`. val = the in-loop
    steering split; NEVER call with 'test' in-loop (that's the sealed final evaluator only)."""
    import data

    _parse = _parse_go
    n = max(1, args.eval_subset_size)
    raws = _sft_smoke_dataset()[:n] if args.smoke else []
    if not args.smoke:
        for ex in _stream_real_examples(args, SFT_DATA_REPO, split):
            raws.append(ex)
            if len(raws) >= n:
                break
    from eval_targets.base import prompt_variant_suffix

    # Empty unless SENPAI_EVAL_PROMPT_VARIANT asks for a variant, so the default val prompt stays
    # byte-identical to the one the model was trained on.
    suffix = prompt_variant_suffix()

    recs = []
    for i, r in enumerate(raws):
        formatted = data.format_cafa5_for_protein_llm(r)
        gt = _parse(r.get("go_mf", "[]")) | _parse(r.get("go_bp", "[]")) | _parse(r.get("go_cc", "[]"))
        recs.append({"protein_id": r.get("protein_id") or f"smoke{i}",
                     "user_turn": _append_prompt_suffix(formatted["prompt"][0], suffix),
                     "sequence": formatted["protein_sequences"][0] if multimodal else None,
                     "gt_terms": gt})
    return recs


def _fuse_ids(model, ids):
    """Embed token ids and scatter ESM2-projected protein (+ optional GO) tokens into the pad positions
    — exactly ProteinLLMModel.forward's fusion, built from public pieces + module-level fuse_embeddings
    (no edits to the protected bioreason_pro/). Grad flows through the trainable projection/LoRA; ESM2
    stays frozen. `model._mm_seq` holds the current protein sequence (set by the callers below)."""
    from bioreason_pro import model as M

    text_embeds = model.text_model.get_input_embeddings()(ids)
    proj = model.protein_projection
    esm = model.protein_encoder.encode_sequences([model._mm_seq], [0], 1)[0]
    protein_flat = proj(esm.to(device=proj[0].weight.device, dtype=proj[0].weight.dtype))
    go_flat = None
    if getattr(model, "go_encoder", None) is not None:
        reduced = model.go_encoder("all")
        if model.go_projection is not None:
            gp = model.go_projection
            reduced = gp(reduced.to(device=gp[0].weight.device, dtype=gp[0].weight.dtype))
        go_flat = reduced
    return M.fuse_embeddings(text_embeds, ids, model.protein_token_id, model.go_token_id,
                            protein_embeds_flat=protein_flat, go_embeds_flat=go_flat)


def _generation_ready(model):
    """Disable gradient checkpointing + enable the KV cache for a no-grad generate() (GC only matters
    for the backward pass; leaving it on forces use_cache=False → O(L^2) recompute). Returns a restore
    fn to flip both back for the training forward."""
    tm = model.text_model
    try:
        tm.gradient_checkpointing_disable()
    except Exception:
        pass
    try:
        tm.config.use_cache = True
    except Exception:
        pass

    def _restore():
        try:
            tm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
        try:
            tm.config.use_cache = False
        except Exception:
            pass

    return _restore


def _fused_generate(model, tok, prompt_text: str, seq: str, args: RunArgs) -> str:
    """Protein-aware greedy decode through the fusion (eval path). Returns the decoded text."""
    import torch

    model._mm_seq = seq
    dev = next(model.text_model.parameters()).device
    input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(dev)
    attn = torch.ones_like(input_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # With inputs_embeds, generate() returns only the newly-decoded tokens (no prompt ids to prepend).
    restore = _generation_ready(model)  # GC off + KV cache on for the no-grad decode
    try:
        gen = model.text_model.generate(inputs_embeds=_fuse_ids(model, input_ids), attention_mask=attn,
                                        max_new_tokens=args.max_completion_length, do_sample=False,
                                        use_cache=True, pad_token_id=pad_id)
    finally:
        restore()
    return tok.decode(gen[0], skip_special_tokens=True)


def _generation_eval(model, tok, args: RunArgs, multimodal: bool, split: str = "val",
                     publish_weave_eval: bool = False) -> dict:
    """Generate over `split` and score a REAL threshold-swept IA-weighted F_max via the sealed
    eval.score_generations. Multimodal → protein-aware fused decode; text → plain LM generate.

    `publish_weave_eval` additionally publishes the run as a weave.Evaluation for per-protein
    inspection. Off by default: in-loop val runs every eval interval and does not need the extra
    round trips, while standalone eval runs do."""
    import torch

    import data
    import eval as sealed_eval

    recs = _eval_prompt_records(args, split, multimodal)
    template = data.CHAT_TEMPLATE_PATH.read_text()
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    records, was_training = [], model.training
    model.eval()
    with torch.no_grad():
        for rec in recs:
            if multimodal:
                text = tok.apply_chat_template([rec["user_turn"]], tokenize=False,
                                               add_generation_prompt=True, chat_template=template)
                text = data.expand_pad_tokens(text, rec["sequence"])
                resp = _fused_generate(model, tok, text, rec["sequence"], args)
            else:
                user_text = next(c["text"] for c in rec["user_turn"]["content"] if c.get("type") == "text")
                text = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                               tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt").to(dev)
                gen = model.generate(**enc, max_new_tokens=args.max_completion_length,
                                     do_sample=False, pad_token_id=pad_id)
                resp = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            records.append({"protein_id": rec["protein_id"], "generated_response": resp,
                            "gt_terms": rec["gt_terms"]})
    if was_training:
        model.train()
    try:
        # diagnostics on the val (steering) path only: coverage and prediction shape explain a low
        # weighted_fmax, and the sealed path must keep emitting exactly the metric of record.
        metrics = sealed_eval.score_generations(
            records,
            f"outputs/eval-{args.wandb_name or 'run'}",
            diagnostics=(split == "val"),
        )
    except Exception as e:
        print(f"[eval] score_generations failed ({type(e).__name__}: {e}); reporting 0.0", flush=True)
        return {"weighted_fmax": 0.0}

    # ADR-014: a score with no readable reasoning behind it is not the deliverable. These travel
    # with weighted_fmax so an arm that wins on F_max while emitting empty traces is visible as
    # such rather than looking like an unqualified win. Reporting only — never fed to training.
    metrics = {**dict(metrics), **_reasoning_diagnostics(records)}

    if publish_weave_eval:
        # Reporting only — `metrics` above is already the authoritative result and is attached to
        # the Evaluation rather than recomputed from per-example scorers.
        from bioreason_pro import go_obo, weave_eval
        from eval_targets.base import active_prompt_variant

        url = weave_eval.publish_evaluation(
            records,
            metrics,
            obo_ancestors=go_obo.load_go_ancestors(sealed_eval.OBO),
            ia=go_obo.load_ia_weights(sealed_eval.IA),
            name=f"{split}-{args.wandb_name or 'run'}-{active_prompt_variant()}",
            attributes={
                "split": split,
                "prompt_variant": active_prompt_variant(),
                "stage": args.stage,
                "esm_layer": args.esm_layer,
                "max_completion_length": args.max_completion_length,
            },
        )
        if url:
            print(f"WEAVE-EVALUATION: {url}", flush=True)
    return metrics


def _log_and_emit_eval(model, tok, args: RunArgs, wandb_run_id: str, multimodal: bool) -> None:
    """Run the val eval (text or protein-aware multimodal), log val_primary/* to W&B, emit SENPAI-RESULT.
    test_primary is reported as the val number (a placeholder — the sealed claim is a separate,
    independent re-run of eval.py --split test on the merged checkpoint; never peek at test in-loop)."""
    metrics = _generation_eval(model, tok, args, multimodal, split="val")
    v = float(metrics.get("weighted_fmax", 0.0))
    if wandb_run_id:
        try:
            import wandb
            wandb.log({f"val_primary/{k}": val for k, val in metrics.items()})
        except Exception:
            pass
    print(f"[eval] val_primary/weighted_fmax={v:.4f} "
          f"(test_primary is a val-proxy; run eval.py --split test for the sealed number)", flush=True)
    emit_result(wandb_run_id, v, v)


def _dist_setup():
    """Init torch.distributed from torchrun/SLURM env (nccl). Returns (rank, world_size, local_rank,
    is_main). Single-process when WORLD_SIZE<=1 (no-op).

    Explicit generous timeout (ADR-036/037 incident): PyTorch's NCCL default is 10 minutes. Multimodal
    RL's per-step collective (the manual grad all-reduce) only runs after EVERY rank sequentially
    generates proteins_per_step*num_generations rollouts, each up to max_completion_length tokens — at
    the post-ADR-037 recommended scale (4*8*3072) a slower rank (an unlucky long-generating protein,
    or an atypically low-EOS-rate step) can genuinely exceed 10 minutes before reaching its own
    all_reduce call, which two real jobs (1681, 1682) hit: the watchdog's timeout-triggered "corrupted
    data" state then surfaced as a FloatingPointError several calls later, masking the real cause. A
    much longer window costs nothing for SFT (no per-step generation, already fast) or for a
    genuinely-hung run (still eventually times out) — it only removes a false floor under RL's now
    much larger per-step generation workload.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import datetime

        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=datetime.timedelta(minutes=90),
            )
    return rank, world, local, rank == 0


def _dist_barrier(world: int) -> None:
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def _enable_gradient_checkpointing(model, args: RunArgs) -> None:
    """Enable gradient checkpointing on the text backbone (trade compute for memory — needed for the
    4B model + long sequences). No-op if disabled or unsupported."""
    if not args.gradient_checkpointing:
        return
    tm = model.text_model if args.use_multimodal else model
    try:
        if hasattr(tm, "config"):
            tm.config.use_cache = False
        if hasattr(tm, "enable_input_require_grads"):
            tm.enable_input_require_grads()
        tm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("train: gradient checkpointing enabled", flush=True)
    except Exception as e:
        print(f"train: gradient checkpointing not enabled ({type(e).__name__}: {e})", flush=True)


def run_sft(
    args: RunArgs,
    deadline,
    wandb_run_id: str = "",
    ckpt_version: str = "",
    wandb_run=None,
) -> int:
    """Supervised fine-tuning (paper Stage 1 text / Stage 2 fusion). Self-contained manual loop over
    ProteinLLMModel.forward(labels=...) — no generation, so it works for the multimodal fusion path
    (unlike RL). LoRA on the text backbone (SFT r=128/α=256 via --lora_r/--lora_alpha), ESM2 frozen,
    micro-batch=1 protein with gradient accumulation, timeout-safe, saves the adapter.

    NOTE: in-loop val eval + the SENPAI-RESULT claim marker need model GENERATION (eval.evaluate is a
    stub; multimodal decode is the pending protein-aware-vLLM step). Until then this trains + logs
    loss/grad-norm; a separate, independent run of eval.py on the merged checkpoint provides the metric."""
    from pathlib import Path

    import torch
    import wandb
    from peft import LoraConfig, get_peft_model

    import data
    from bioreason_pro import model as M

    rank, world, local, is_main = _dist_setup()
    cuda_device = torch.device("cuda", local) if torch.cuda.is_available() else None
    dev = cuda_device or torch.device("cpu")
    run_started = time.monotonic()
    if cuda_device is not None:
        # The current device is set by _dist_setup for DDP and defaults to cuda:0 for a single
        # process. Passing an explicit device is rejected by some recent CUDA allocator builds.
        torch.cuda.reset_peak_memory_stats()
    if args.use_multimodal:
        model, tok = build_multimodal_policy(args)          # LoRA applied to text_model inside
    else:
        model, tok = M.build_text_model(M.ModelConfig(use_multimodal=False), args.model_name or None)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES))
    model = model.to(dev)
    model.train()
    _enable_gradient_checkpointing(model, args)
    core = model                                            # unwrapped (for save/eval/fusion helpers)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local] if cuda_device is not None else None,
                    find_unused_parameters=True, broadcast_buffers=False)
    template = data.CHAT_TEMPLATE_PATH.read_text()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate)
    accum = max(1, args.gradient_accumulation_steps)
    max_steps = args.resolved_max_steps()
    if is_main:
        print(f"SFT: world={world} dev={dev} multimodal={args.use_multimodal} "
              f"trainable={sum(p.numel() for p in trainable):,} accum={accum} "
              f"max_steps={max_steps}", flush=True)

    def _valid_kw():
        """Infinite per-rank stream of valid forward-kwargs (re-streams its shard, skips oversized/empty)
        so every rank does exactly the same #backwards per step → DDP stays in lockstep (no NCCL hang)."""
        while True:
            src = (_sft_smoke_dataset() if args.smoke
                   else _stream_real_examples(args, SFT_DATA_REPO, "rl_train", rank, world))
            yielded = False
            for ex in src:
                kw = _sft_forward_inputs(ex, tok, template, dev, args.use_multimodal, args.max_seq_len,
                                         mask_empty_think=args.mask_empty_think)
                if kw is not None:
                    yielded = True
                    yield kw
            if not yielded:
                raise RuntimeError("no valid SFT examples for this rank (all skipped by max_seq_len?)")

    it = _valid_kw()
    opt.zero_grad()
    examples_seen = 0
    tokens_seen = 0
    for gstep in range(1, max_steps + 1):
        running = 0.0
        for _ in range(accum):
            batch = next(it)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"SFT step {gstep}: non-finite loss {float(loss)}")
            (loss / accum).backward()
            running += float(loss.detach())
            examples_seen += int(batch["input_ids"].shape[0])
            tokens_seen += int(batch["attention_mask"].sum().item())
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"SFT step {gstep}: non-finite gradient norm {float(gnorm)}")
        opt.step()
        opt.zero_grad()
        if is_main:
            avg = running / accum
            elapsed = max(time.monotonic() - run_started, 1e-9)
            telemetry = {
                "sft/loss": avg,
                "sft/grad_norm": float(gnorm),
                "sft/step": gstep,
                "sft/examples_per_second": examples_seen / elapsed,
                "sft/tokens_per_second": tokens_seen / elapsed,
                "system/elapsed_seconds": elapsed,
            }
            if cuda_device is not None:
                telemetry.update({
                    "system/gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "system/gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "system/gpu_peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                })
            if wandb_run_id:
                try:
                    wandb.log(telemetry)
                except Exception:
                    pass
            print(
                f"SFT step {gstep} loss={avg:.4f} grad_norm={float(gnorm):.3f} "
                f"tokens/s={telemetry['sft/tokens_per_second']:.1f}",
                flush=True,
            )
        # Lockstep deadline: rank 0 decides, broadcast so all ranks stop together.
        stop = 1 if (deadline is not None and time.monotonic() > deadline) else 0
        if world > 1:
            import torch.distributed as dist
            t = torch.tensor([stop], device=dev)
            dist.broadcast(t, src=0)
            stop = int(t.item())
        if stop:
            break

    # Tear down the process group BEFORE the (long, rank-0-only) eval. Otherwise ranks 1..N-1 block on
    # a barrier while rank 0 generates, and the NCCL barrier's ~10-min timeout aborts them mid-eval.
    _dist_barrier(world)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    if is_main:
        out_dir = args.output_dir or f"outputs/sft-{args.wandb_name or 'run'}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (core.text_model if args.use_multimodal else core).save_pretrained(out_dir)
        print(f"SFT: saved LoRA adapter to {out_dir}", flush=True)
        if args.use_multimodal:
            _save_mm_extras(core, args, out_dir)   # persist trained projections + build args for eval reload
        else:
            _save_data_manifest(args, out_dir)
        # Real val eval → val_primary/weighted_fmax + SENPAI-RESULT (rank 0 only; single-GPU generate).
        _log_and_emit_eval(core, tok, args, wandb_run_id, multimodal=args.use_multimodal)
        _log_checkpoint_artifact(out_dir, args, wandb_run)
        if args.smoke:
            print("SFT-SMOKE-OK", flush=True)
    return 0


def _deadline_callback(deadline):
    """A TrainerCallback that stops training before the SENPAI wall-clock deadline (timeout-safe)."""
    from transformers import TrainerCallback

    class Deadline(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if deadline is not None and time.monotonic() > deadline:
                control.should_training_stop = True
            return control

    return Deadline()


def _load_reward(args: RunArgs, wandb_run_id: str, ckpt_version: str):
    """Build the traced composite reward from verified assets, or no assets for a smoke run."""
    from bioreason_pro import go_obo, weave_tracing as WT
    from bioreason_pro.license_policy import validate_reference_assets

    if args.smoke:
        obo, ia = {}, None
    else:
        validate_reference_assets(use="reward")
        obo = go_obo.load_go_ancestors("data/go-basic.obo")
        ia = go_obo.load_ia_weights("data/IA.txt")
    ctx = WT.RolloutCtx(stage="rl", wandb_run_id=wandb_run_id, ckpt_version=ckpt_version,
                        rl_algo=args.rl_algo,
                        importance_sampling_level="sequence" if args.rl_algo == "gspo" else "token")
    return WT.make_traced_reward_fn(obo, ia, ctx)


def _last_logged(trainer, key: str, default: float = 0.0) -> float:
    """Most recent value of `key` from the trainer's log history (e.g. the final mean reward)."""
    for entry in reversed(trainer.state.log_history):
        if key in entry:
            return float(entry[key])
    return default


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _merge_lora_adapter(text_model, adapter_dir: str):
    """Merge one PEFT adapter and return the resulting plain text model."""
    from peft import PeftModel

    import peft.utils.save_and_load as save_load

    save_load._maybe_shard_state_dict_for_tp = lambda state_dict, *a, **k: state_dict
    loaded = PeftModel.from_pretrained(text_model, adapter_dir)
    n_lora = sum(1 for key in loaded.state_dict() if "lora_" in key)
    merged = loaded.merge_and_unload()
    print(f"[policy] merged adapter from {adapter_dir} ({n_lora} LoRA tensors)", flush=True)
    return merged


def _load_projection_state(core, checkpoint_dir: str) -> None:
    """Load the trained protein/GO projections from a validated checkpoint."""
    from pathlib import Path

    import torch

    extras = torch.load(
        Path(checkpoint_dir) / "projections.pt",
        map_location="cpu",
        weights_only=True,
    )
    core.protein_projection.load_state_dict(extras["protein_projection"])
    if core.go_projection is not None and extras.get("go_projection") is not None:
        core.go_projection.load_state_dict(extras["go_projection"])
    print(f"[policy] loaded projections from {checkpoint_dir}", flush=True)


def build_multimodal_policy(args: RunArgs):
    """Assemble the allowlisted Qwen3 + frozen ESM2 fusion policy and LoRA the text backbone.

    The protein encoder name is checked against ``approved_assets.json`` before model loading;
    projections remain full-rank trainable modules.
    """
    from peft import LoraConfig, get_peft_model

    from bioreason_pro import model as M

    args.validate()
    mcfg = M.ModelConfig(
        text_model_name=args.model_name or M.ModelConfig.text_model_name,
        esm_model_name=args.esm_model_name, esm_layer=args.esm_layer, freeze_esm=args.freeze_esm,
        use_multimodal=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout)
    fused = M.build_model(mcfg, args.model_name or None)
    # SFT→RL hand-off: the active RL run resolved this directory through use_artifact. Merge the SFT
    # LoRA into the base, THEN train a fresh RL LoRA (RL does not continue the SFT adapter itself).
    if args.sft_adapter:
        _load_projection_state(fused, args.sft_adapter)
        fused.text_model = _merge_lora_adapter(fused.text_model, args.sft_adapter)
    fused.text_model = get_peft_model(fused.text_model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES))
    return fused, fused.text_tokenizer


# ----------------------------------------------------------------- sealed-test evaluation
# The in-loop val eval (_generation_eval) runs on the LIVE in-memory model, so its projections are the
# trained ones. A STANDALONE sealed-test eval must instead RELOAD a checkpoint from disk — which means
# the trained projections + the exact build args have to be persisted (see _save_mm_extras), then
# reconstructed here. This is what eval.py:evaluate() delegates to (target-agnostic; datasets live in
# the editable eval_targets/ package, scoring stays in the protected eval.score_generations).

def _save_mm_extras(core, args: RunArgs, out_dir: str) -> None:
    """Persist the trained projections + a run-args snapshot next to the saved LoRA adapter.

    The adapter save covers `text_model` (the Qwen LoRA) only; `protein_projection`/`go_projection`
    live on the ProteinLLMModel wrapper and train full-rank. Without this, a standalone reload
    (evaluate_checkpoint) rebuilds them at RANDOM init → the protein/GO fusion is garbage → the
    reported test F_max is meaningless. run_args.json lets eval rebuild the exact architecture
    (esm_model_name, esm_layer, lora_*) without re-specifying it."""
    import json
    import shutil
    from pathlib import Path

    import torch
    from bioreason_pro.license_policy import approved_model_revision, load_approved_assets

    extras = {"protein_projection": core.protein_projection.state_dict()}
    if getattr(core, "go_projection", None) is not None:
        extras["go_projection"] = core.go_projection.state_dict()
    torch.save(extras, Path(out_dir) / "projections.pt")
    snapshot = {
        **vars(args),
        "asset_manifest_schema_version": load_approved_assets()["schema_version"],
        "text_model_revision": approved_model_revision(
            args.model_name or "Qwen/Qwen3-4B-Thinking-2507", "text"
        ),
        "esm_model_revision": approved_model_revision(args.esm_model_name, "protein"),
    }
    if args.stage == "rl":
        from bioreason_pro.license_policy import require_immutable_wandb_artifact_ref

        resolved = require_immutable_wandb_artifact_ref(
            args.resolved_sft_artifact,
            "resolved SFT input artifact",
        )
        if resolved != args.sft_artifact:
            raise ValueError("RL checkpoint metadata must pin the SFT artifact used by this run")
        source = Path(args.sft_adapter)
        bundled = Path(out_dir) / "sft_adapter"
        if bundled.exists():
            raise FileExistsError(f"{bundled}: refuse to reuse a stale bundled SFT checkpoint")
        shutil.copytree(source, bundled)
        snapshot["sft_artifact"] = resolved
        snapshot["resolved_sft_artifact"] = resolved
        snapshot["sft_adapter"] = "sft_adapter"
    (Path(out_dir) / "run_args.json").write_text(
        json.dumps(snapshot, default=str, indent=2)
    )
    _save_data_manifest(args, out_dir)
    print(f"[save] wrote projections.pt ({', '.join(extras)}) + run_args.json to {out_dir}", flush=True)


def _save_data_manifest(args: RunArgs, out_dir: str) -> None:
    """Persist the exact row source, prompt fields, label fields, and split rule."""
    from pathlib import Path

    from bioreason_pro.data_contract import data_contract_json

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "data_manifest.json").write_text(
        data_contract_json(args.stage, args.smoke, args.seed) + "\n", encoding="utf-8"
    )


def _qualified_artifact_ref(artifact, run) -> str:
    """Return and validate the immutable fully-qualified reference of a W&B Artifact."""
    from bioreason_pro.license_policy import require_immutable_wandb_artifact_ref

    qualified = getattr(artifact, "qualified_name", "") or ""
    if qualified:
        return require_immutable_wandb_artifact_ref(str(qualified))
    name = str(getattr(artifact, "name", "") or "")
    version = str(getattr(artifact, "version", "") or "")
    if ":" in name:
        name, name_version = name.rsplit(":", 1)
        version = version or name_version
    reference = f"{run.entity}/{run.project}/{name}:{version}"
    return require_immutable_wandb_artifact_ref(reference)


def use_model_artifact(run, artifact_ref: str, download_root: str) -> tuple[str, str]:
    """Declare a model input on ``run``, download it, and return (local_dir, immutable_ref).

    Calling ``run.use_artifact`` here is the formal model hand-off: W&B records the input edge from
    the artifact to this run. ``wandb.Api().artifact`` must not replace it because that read would
    not create run lineage.
    """
    from pathlib import Path

    from bioreason_pro.license_policy import (
        require_immutable_wandb_artifact_ref,
        validate_checkpoint_layout,
    )

    requested = require_immutable_wandb_artifact_ref(artifact_ref)
    target = Path(download_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    artifact = run.use_artifact(requested, type="model")
    local_dir = str(Path(artifact.download(root=str(target))).resolve())
    resolved = _qualified_artifact_ref(artifact, run)
    if resolved != requested:
        raise RuntimeError(
            f"W&B resolved {requested!r} to {resolved!r}; immutable input versions must match"
        )
    validate_checkpoint_layout(local_dir)
    print(f"[artifact] use_artifact {resolved} -> {local_dir}", flush=True)
    return local_dir, resolved


def _validate_sft_input_architecture(args: RunArgs, checkpoint_dir: str) -> None:
    """Fail before model loading when the RL launcher disagrees with its SFT parent."""
    from bioreason_pro.license_policy import validate_checkpoint_layout
    from bioreason_pro.model import ModelConfig

    saved = validate_checkpoint_layout(checkpoint_dir)
    requested_text = args.model_name or ModelConfig.text_model_name
    saved_text = saved.get("model_name") or ModelConfig.text_model_name
    if requested_text != saved_text:
        raise ValueError(
            f"SFT artifact text model {saved_text!r} does not match requested "
            f"{requested_text!r}"
        )
    if args.esm_model_name != saved.get("esm_model_name"):
        raise ValueError(
            f"SFT artifact protein model {saved.get('esm_model_name')!r} does not match requested "
            f"{args.esm_model_name!r}"
        )
    if args.esm_layer != saved.get("esm_layer"):
        raise ValueError(
            f"SFT artifact ESM layer {saved.get('esm_layer')!r} does not match requested "
            f"{args.esm_layer!r}"
        )


def _resolve_sft_artifact(args: RunArgs, run) -> None:
    """Resolve the SFT input once on rank 0 and share its validated local path with torchrun ranks."""
    if args.stage != "rl" or not args.use_multimodal:
        return

    import hashlib
    from pathlib import Path

    rank = int(os.environ.get("RANK", "0"))
    runtime_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else Path(os.environ.get("BIOREASON_ARTIFACT_ROOT", "outputs/wandb-artifacts"))
    )
    digest = hashlib.sha256(args.sft_artifact.encode("utf-8")).hexdigest()[:12]
    session = hashlib.sha256(
        (args.output_dir or os.environ.get("SLURM_JOB_ID", "local")).encode("utf-8")
    ).hexdigest()[:8]
    collection = args.sft_artifact.rsplit("/", 1)[-1].replace(":", "-")
    target = (runtime_root / f"{collection}-{digest}-{session}").resolve()
    ready = target.parent / f".{target.name}.ready.json"
    failed = target.parent / f".{target.name}.error.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        if run is None:
            raise RuntimeError("rank 0 needs an active W&B run to call use_artifact")
        ready.unlink(missing_ok=True)
        failed.unlink(missing_ok=True)
        try:
            local_dir, resolved = use_model_artifact(run, args.sft_artifact, str(target))
            _validate_sft_input_architecture(args, local_dir)
            payload = {"path": local_dir, "artifact_ref": resolved}
            tmp = ready.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, ready)
        except Exception as exc:
            failed.write_text(
                json.dumps({"type": type(exc).__name__, "message": str(exc)}),
                encoding="utf-8",
            )
            raise
    else:
        deadline = time.monotonic() + 30 * 60
        while not ready.is_file():
            if failed.is_file():
                error = json.loads(failed.read_text(encoding="utf-8"))
                raise RuntimeError(f"rank 0 artifact download failed: {error}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for {ready}")
            time.sleep(1)

    receipt = json.loads(ready.read_text(encoding="utf-8"))
    args.sft_adapter = receipt["path"]
    args.resolved_sft_artifact = receipt["artifact_ref"]
    args.sft_artifact = receipt["artifact_ref"]
    args.validate(require_training_dataset=False)


def _log_checkpoint_artifact(out_dir: str, args: RunArgs, run) -> str:
    """Publish a complete model output and wait until its immutable W&B version is finalized."""
    if run is None:
        return ""
    from pathlib import Path

    import wandb

    receipt_path = Path(out_dir) / "wandb_artifact.json"
    receipt_path.unlink(missing_ok=True)
    artifact = wandb.Artifact(
        name=f"bioreasonpro-{args.stage}-checkpoint",
        type="model",
        metadata={
            "stage": args.stage,
            "smoke": args.smoke,
            "model_name": args.model_name or "Qwen/Qwen3-4B-Thinking-2507",
            "esm_model_name": args.esm_model_name if args.use_multimodal else None,
            "producer_run_id": run.id,
            "input_sft_artifact": args.resolved_sft_artifact or None,
        },
    )
    artifact.add_dir(out_dir)
    logged = run.log_artifact(artifact, aliases=["latest", args.stage])
    logged.wait()
    reference = _qualified_artifact_ref(logged, run)
    receipt = {
        "schema_version": 1,
        "artifact_ref": reference,
        "artifact_type": "model",
        "stage": args.stage,
        "producer_run": f"{run.entity}/{run.project}/{run.id}",
        "input_sft_artifact": args.resolved_sft_artifact or None,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run.summary[f"{args.stage}_artifact_ref"] = reference
    print(f"WANDB-ARTIFACT: {reference}", flush=True)
    return reference


def _load_run_args(checkpoint_dir: str, overrides: dict | None = None) -> RunArgs:
    """Rebuild RunArgs from required, policy-validated checkpoint metadata."""
    from dataclasses import fields
    from pathlib import Path

    from bioreason_pro.license_policy import validate_checkpoint_layout

    saved = validate_checkpoint_layout(checkpoint_dir)
    valid = {f.name for f in fields(RunArgs)}
    kw = {k: v for k, v in {**saved, **(overrides or {})}.items() if k in valid}
    args = RunArgs(**kw)
    if args.stage == "rl" and args.sft_adapter == "sft_adapter":
        args.sft_adapter = str(Path(checkpoint_dir).resolve() / "sft_adapter")
    args.validate(require_training_dataset=False)
    return args


def evaluate_checkpoint(checkpoint_dir: str, args: RunArgs):
    """Assemble the fusion model, load the trained LoRA adapter + projections from `checkpoint_dir`,
    and return (model, tok) ready for _fused_generate. Mirrors build_multimodal_policy's assembly +
    adapter merge, but does NOT wrap a fresh training LoRA. Supports the PEFT-adapter dir layout only;
    full-model checkpoints and checkpoints without provenance metadata are intentionally rejected."""
    from bioreason_pro.license_policy import validate_checkpoint_layout

    validate_checkpoint_layout(
        checkpoint_dir,
        expected_text_model=args.model_name or "Qwen/Qwen3-4B-Thinking-2507",
        expected_protein_model=args.esm_model_name,
    )
    args.validate(require_training_dataset=False)

    import os

    import torch

    from bioreason_pro import model as M

    if not args.use_multimodal:
        raise NotImplementedError("evaluate_checkpoint supports the multimodal fusion path only")
    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"{checkpoint_dir}: no adapter_config.json — full-model (non-LoRA) checkpoints are not "
            f"supported yet (see plan follow-ups)")

    mcfg = M.ModelConfig(
        text_model_name=args.model_name or M.ModelConfig.text_model_name,
        esm_model_name=args.esm_model_name, esm_layer=args.esm_layer, freeze_esm=args.freeze_esm,
        use_multimodal=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout)
    fused = M.build_model(mcfg, args.model_name or None)

    if args.stage == "rl":
        fused.text_model = _merge_lora_adapter(
            fused.text_model, os.path.join(checkpoint_dir, "sft_adapter")
        )
    fused.text_model = _merge_lora_adapter(fused.text_model, checkpoint_dir)
    _load_projection_state(fused, checkpoint_dir)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fused = fused.to(dev)
    fused.eval()
    return fused, fused.text_tokenizer


def run_sealed_eval(checkpoint_dir: str, target: str = "bioreason_pro_test", split: str = "test",
                    subset_size: int | None = None, max_completion_length: int = 1024,
                    overrides: dict | None = None, *, shard_index: int = 0,
                    num_shards: int = 1) -> list[dict]:
    """Reload `checkpoint_dir`, stream the held-out `target`, protein-aware greedy-decode each row, and
    return records [{protein_id, generated_response, gt_terms}] for eval.score_generations.

    `split` is accepted for CLI parity; the actual row selection is the target's split_mode (the
    `bioreason_pro_test` target = the whole sealed `test` split). HF greedy fused decode only — there
    is no vLLM path for the fusion. `max_completion_length` must be large enough (~1024+) that the GO
    answer after </think> isn't truncated (the RunArgs default of 256 truncates it → empty predictions)."""
    import torch

    import data
    import eval_targets

    tgt = eval_targets.get_target(target)
    args = _load_run_args(checkpoint_dir, overrides)
    args.use_multimodal = True
    args.max_completion_length = max_completion_length
    model, tok = evaluate_checkpoint(checkpoint_dir, args)
    template = data.CHAT_TEMPLATE_PATH.read_text()
    records: list[dict] = []
    with torch.no_grad():
        for rec in eval_targets.stream_eval_records(
            tgt,
            seed=args.seed,
            limit=subset_size,
            shard_index=shard_index,
            num_shards=num_shards,
        ):
            text = tok.apply_chat_template([rec["user_turn"]], tokenize=False,
                                           add_generation_prompt=True, chat_template=template)
            text = data.expand_pad_tokens(text, rec["sequence"])
            resp = _fused_generate(model, tok, text, rec["sequence"], args)
            records.append({"protein_id": rec["protein_id"], "generated_response": resp,
                            "gt_terms": rec["gt_terms"]})
            if len(records) % 50 == 0:
                print(f"[eval] {target}: generated {len(records)} responses...", flush=True)
    print(
        f"[eval] {target}: generated {len(records)} responses "
        f"(split={split}, shard={shard_index + 1}/{num_shards})",
        flush=True,
    )
    return records


def _multimodal_build_smoke(args: RunArgs, wandb_run_id: str = "") -> int:
    """SENPAI_MM_BUILD=1 hook: assemble the ESM2/GO fusion policy and run ONE fwd/bwd on a synthetic
    single-protein batch — validates build_model + the scatter-add fusion + finite grads on GPU (no RL
    rollout). Not an experiment: prints a MM-BUILD marker instead of the SENPAI-RESULT claim marker."""
    import torch

    import data

    model, tok = build_multimodal_policy(args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MM-BUILD: assembled fusion policy on {dev} | params={n_params:,} trainable={n_train:,}",
          flush=True)

    # One fwd/bwd on a synthetic single protein. render_chat expands protein pads to min(len,1024)+2,
    # matching ESM2 encode_sequences' per-residue count (the scatter-add shape contract).
    try:
        seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        example = {"prompt": {"system": "You are a protein function prediction assistant.",
                              "user": "Predict the GO terms for this protein.",
                              "assistant_answer": "GO:0005524", "assistant_reasoning": "It binds ATP."},
                   "sequence": seq}
        enc = tok(data.render_chat(example, tok), return_tensors="pt")
        input_ids, attn = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        out = model(input_ids=input_ids, attention_mask=attn,
                    protein_sequences=[seq], batch_idx_map=[0], labels=input_ids)
        out.loss.backward()
        n_prot = int((input_ids == model.protein_token_id).sum().item())
        finite = bool(torch.isfinite(out.loss).item())
        print(f"MM-BUILD: fwd/bwd OK loss={float(out.loss):.4f} finite={finite} "
              f"protein_pad_tokens={n_prot}", flush=True)
    except Exception as e:  # build succeeded; report the forward diagnostic without failing the smoke
        print(f"MM-BUILD: assembly OK but fwd/bwd raised: {type(e).__name__}: {e}", flush=True)
    return 0


def _mm_prompt_text(tok, template: str, user_turn: dict, seq: str) -> str:
    """Render a multimodal generation prompt (user turn + generation prompt, protein/GO pads expanded)."""
    import data

    text = tok.apply_chat_template([user_turn], tokenize=False, add_generation_prompt=True,
                                   chat_template=template)
    return data.expand_pad_tokens(text, seq)


def _gen_debug(model, tok, prompt_text: str, seq: str) -> None:
    """SENPAI_GEN_DEBUG: inspect the fused policy's next-token distribution + compare MANUAL greedy vs
    tight-sampled (top_k) vs full-sampled decode — to tell an under-trained (base-like) distribution
    from a generate() sampling bug. Uses the same fused prompt forward + KV-cache continuation."""
    import torch

    _generation_ready(model)  # GC off + KV cache on so the manual past_key_values loop is valid
    model._mm_seq = seq
    dev = next(model.text_model.parameters()).device
    ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(dev)

    def _prompt_fwd():
        with torch.no_grad():
            o = model.text_model(inputs_embeds=_fuse_ids(model, ids), use_cache=True)
        return o.logits[0, -1].float(), o.past_key_values

    lg0, _ = _prompt_fwd()
    tp, ti = torch.softmax(lg0, -1).topk(15)
    print("[GENDBG] top15@prompt-end: " + " ".join(f"{p:.3f}:{tok.decode([i])!r}"
          for p, i in zip(tp.tolist(), ti.tolist())), flush=True)

    def _decode(sample: bool, top_k: int, temp: float, n: int = 48) -> str:
        lg, past = _prompt_fwd()
        out = []
        with torch.no_grad():
            for _ in range(n):
                step_logits = lg / temp
                if sample and top_k:
                    v, _i = step_logits.topk(top_k)
                    step_logits = step_logits.masked_fill(
                        step_logits < v[-1], float("-inf")
                    )
                nt = (
                    int(torch.multinomial(torch.softmax(step_logits, -1), 1))
                    if sample
                    else int(step_logits.argmax())
                )
                out.append(nt)
                s = model.text_model(input_ids=torch.tensor([[nt]], device=dev),
                                     past_key_values=past, use_cache=True)
                lg, past = s.logits[0, -1].float(), s.past_key_values
        return tok.decode(out)

    print(f"[GENDBG] manual-greedy:        {_decode(False, 0, 1.0)[:220]!r}", flush=True)
    print(f"[GENDBG] manual-topk20-t0.7:   {_decode(True, 20, 0.7)[:220]!r}", flush=True)
    print(f"[GENDBG] manual-fullsample-0.7:{_decode(True, 0, 0.7)[:220]!r}", flush=True)


def _fused_rollout(model, tok, prompt_text: str, seq: str, args: RunArgs):
    """Sample ONE rollout through the fusion. Returns (completion_text, completion_ids [1, T])."""
    import torch

    model._mm_seq = seq
    dev = next(model.text_model.parameters()).device
    ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(dev)
    attn = torch.ones_like(ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    restore = _generation_ready(model)  # GC off + KV cache on for the no-grad rollout
    try:
        with torch.no_grad():
            gen = model.text_model.generate(inputs_embeds=_fuse_ids(model, ids), attention_mask=attn,
                                            max_new_tokens=args.max_completion_length, do_sample=True,
                                            temperature=args.rl_temperature, top_p=0.95,
                                            use_cache=True, pad_token_id=pad_id)
    finally:
        restore()
    return tok.decode(gen[0], skip_special_tokens=True), gen[:, :].detach()


def _fused_completion_logps(model, tok, prompt_text: str, seq: str, comp_ids, args: RunArgs, ref: bool = False):
    """PER-TOKEN log-probs (comp_len,) of `comp_ids` under the fused policy. ref=True disables the RL
    LoRA adapter — recovering the frozen SFT-merged reference policy — and runs under no_grad (for the
    KL anchor). ref=False keeps grad through the RL LoRA + projections (the policy)."""
    import contextlib

    import torch

    model._mm_seq = seq
    dev = next(model.text_model.parameters()).device
    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(dev)
    full_ids = torch.cat([prompt_ids, comp_ids.to(dev)], dim=1)
    attn = torch.ones_like(full_ids)
    comp_len = comp_ids.shape[1]
    tgt = full_ids[:, -comp_len:]
    adapter_ctx = model.text_model.disable_adapter() if ref else contextlib.nullcontext()
    grad_ctx = torch.no_grad() if ref else contextlib.nullcontext()
    with adapter_ctx, grad_ctx:
        logits = model.text_model(inputs_embeds=_fuse_ids(model, full_ids), attention_mask=attn).logits[:, :-1, :]
        lp = torch.log_softmax(logits[:, -comp_len:, :].float() / max(args.rl_temperature, 1e-6), dim=-1)
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]  # (comp_len,)


def _reward_weights_for_variant(reward_variant: str):
    """The RewardWeights each reward variant actually trains with — kept in train.py with the rest
    of the RL wiring (one source of truth, per rewards.py's own REASONED_VARIANTS comment), pulled
    into a named function so this exact configuration can be pinned by a test.

    aspect_mean_reasoned zeroes lambda_fmt/lambda_len (ADR-027, plan.md Phase 6): the paper's own RL
    reward is a single weighted-F_max term scored only against the final answer, with no format or
    conciseness counterpart in the paper or in upstream's code (upstream ships no RL script at all)
    — carrying them here was a local invention against an unresolved "pin from paper" TODO, and
    ADR-027 found there is no paper value to pin. ADR-013 separately found r_format measurably inert
    under GRPO's per-group advantage centering (identical 1.0 on all 472 sampled rollouts), so this
    also drops a term already known to contribute nothing to the gradient it was meant to shape.
    Every other variant (union, aspect_mean, aspect_mean_specific) keeps its exact historical
    arithmetic — this correction is scoped to the reasoned variant Phase 6 actually runs.
    """
    from bioreason_pro import rewards

    if reward_variant == "aspect_mean_specific":
        return rewards.RewardWeights(lambda_spec=0.1)
    if reward_variant in rewards.REASONED_VARIANTS:
        return rewards.RewardWeights(
            lambda_fmt=0.0, lambda_len=0.0, lambda_reason=0.25, lambda_truncation=0.1
        )
    return rewards.RewardWeights()


def _run_multimodal_grpo(
    args: RunArgs,
    deadline,
    wandb_run_id: str = "",
    ckpt_version: str = "",
    wandb_run=None,
) -> int:
    """Minimal DR-GRPO for the multimodal fusion policy (the paper's Stage 3, β=0). Self-contained
    (TRL's rollout+logprob path can't thread a custom protein modality): per step, draw
    proteins_per_step proteins, sample num_generations rollouts of each via the fused decode, score
    with the composite reward, form per-protein group-mean baseline advantages — optionally
    normalized by ONE std pooled across the whole step's batch (ADR-036/037, paper Eq. 4.19-4.20;
    default stays Dr.GRPO's no-std baseline) — and take a policy-gradient step through the fused
    completion log-prob. ESM2 frozen; LoRA (RL r=16/α=32) + projections train. Ends with the
    multimodal val eval + SENPAI-RESULT."""
    from pathlib import Path

    import torch

    import data
    from bioreason_pro import go_obo, rewards, weave_tracing

    rank, world, local, is_main = _dist_setup()
    dev = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    model, tok = build_multimodal_policy(args)
    model = model.to(dev)
    model.train()
    _enable_gradient_checkpointing(model, args)   # memory: 4B backbone + long prompts on one GPU
    template = data.CHAT_TEMPLATE_PATH.read_text()

    if args.smoke:
        obo, ia = {}, None
    else:
        from bioreason_pro.license_policy import validate_reference_assets

        validate_reference_assets(use="reward")
        obo = go_obo.load_go_ancestors("data/go-basic.obo")
        ia = go_obo.load_ia_weights("data/IA.txt")
    reward_ctx = weave_tracing.RolloutCtx(
        stage="rl",
        wandb_run_id=wandb_run_id,
        ckpt_version=ckpt_version,
        rl_algo=args.rl_algo,
        importance_sampling_level="sequence" if args.rl_algo == "gspo" else "token",
    )
    # Aspect-aware reward is opt-in: the shipped union reward is aspect-blind while the metric is an
    # aspect mean, which is what let GRPO trade MF away for CC at no apparent cost.
    reward_variant = rewards.active_reward_variant()
    go_aspects = (
        go_obo.load_go_aspects("data/go-basic.obo")
        if reward_variant in rewards.ASPECT_AWARE_VARIANTS
        else None
    )
    # Only the reasoned variants need names, and loading them costs a second obo parse, so it stays
    # opt-in. Faithfulness scores 0 without them, which would silently zero the term — hence the
    # fail-closed check below rather than a quiet fallback.
    go_names = (
        go_obo.load_go_names("data/go-basic.obo")
        if reward_variant in rewards.REASONED_VARIANTS
        else None
    )
    reward_weights = _reward_weights_for_variant(reward_variant)
    if reward_variant in rewards.REASONED_VARIANTS and not go_names:
        raise ValueError(
            f"reward variant {reward_variant!r} scores reasoning faithfulness against GO names, "
            "but none were loaded — every rollout would score 0 on that term and the variant "
            "would silently degrade to aspect_mean. Check data/go-basic.obo."
        )
    print(f"[rl] reward variant: {reward_variant}"
          f"{' (' + str(len(go_aspects)) + ' terms mapped)' if go_aspects else ''}"
          f"{f' lambda_spec={reward_weights.lambda_spec}' if reward_weights.lambda_spec else ''}"
          f"{f' lambda_reason={reward_weights.lambda_reason} names={len(go_names)}' if go_names else ''}",
          flush=True)
    reward_fn = weave_tracing.make_traced_reward_fn(
        obo, ia, reward_ctx, weights=reward_weights, go_aspects=go_aspects, go_names=go_names
    )
    # GRPO and GSPO differ ONLY in how the importance-sampling ratio is aggregated (per-token vs
    # per-sequence). This loop has no ratio at all: rollouts are generated fresh from the current
    # policy each step and consumed by a single update, so the ratio is identically 1 and the loss is
    # a plain on-policy policy gradient (`-(a/G) * pol.mean()`). Accepting --rl_algo gspo here would
    # run an exact GRPO duplicate while `reward_ctx` stamped every Weave trace
    # importance_sampling_level="sequence" — a fabricated null result for GSPO that nothing downstream
    # could detect. Refuse instead. The text-only path implements gspo for real, via TRL's
    # GRPOConfig(importance_sampling_level=...) in `_grpo_config`.
    if args.rl_algo != "grpo":
        raise ValueError(
            f"--rl_algo {args.rl_algo!r} is not implemented for the multimodal RL path: this loop is "
            "strictly on-policy and has no importance-sampling ratio to aggregate, so gspo would be "
            "indistinguishable from grpo. Use --use_multimodal false for the TRL path that honours "
            "it, or implement a ratio here first."
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.learning_rate)
    G = max(2, args.num_generations)
    B = max(1, args.proteins_per_step)
    max_steps = args.resolved_max_steps()
    if is_main:
        print(f"MM-GRPO: world={world} dev={dev} G={G} B={B} "
              f"advantage_normalization={args.rl_advantage_normalization} "
              f"trainable={sum(p.numel() for p in trainable):,} max_steps={max_steps}",
              flush=True)

    def _prompts():
        """Infinite per-rank stream of (prompt_text, seq, gt_cols) — rank-sharded, re-streamed → lockstep.
        DDP for a custom loop = manual grad all-reduce (below), so every rank must do the same #steps."""
        while True:
            src = (_sft_smoke_dataset() if args.smoke
                   else _stream_real_examples(args, RL_DATA_REPO, "rl_train", rank, world))
            any_ex = False
            for ex in src:
                formatted = ex if isinstance(ex.get("prompt"), list) else data.format_cafa5_for_protein_llm(ex)
                seq = formatted["protein_sequences"][0]
                any_ex = True
                yield (ex.get("protein_id", ""), _mm_prompt_text(
                           tok, template, formatted["prompt"][0], seq
                       ), seq,
                       {c: ex.get(c, "[]") for c in ("go_mf", "go_bp", "go_cc")})
            if not any_ex:
                raise RuntimeError("no RL prompts for this rank")

    it = _prompts()
    for gstep in range(1, max_steps + 1):
        protein_id, prompt_text, seq, gt = next(it)
        if os.environ.get("SENPAI_GEN_DEBUG") == "1" and (world <= 1 or is_main):
            _gen_debug(model, tok, prompt_text, seq)
            return 0

        # ---- Phase 1: collect B proteins x G rollouts each. Nothing backward yet — the pooled std
        # below (when enabled) needs every protein's rewards before any advantage can be computed.
        groups = []
        for b in range(B):
            if b > 0:
                protein_id, prompt_text, seq, gt = next(it)
            texts, comp_ids = [], []
            for _g in range(G):
                t, cid = _fused_rollout(model, tok, prompt_text, seq, args)
                texts.append(t)
                comp_ids.append(cid)
            reward_columns = {
                **{key: [value] * G for key, value in gt.items()},
                "protein_id": [protein_id] * G,
                "rl_global_step": [gstep] * G,
            }
            reward_values = reward_fn([prompt_text] * G, texts, **reward_columns)
            rewards.validate_reward_group(reward_values, require_non_degenerate=False)
            r_b = torch.tensor(reward_values, dtype=torch.float32)
            true_go = set()
            for column in ("go_mf", "go_bp", "go_cc"):
                true_go |= _parse_go(gt.get(column, "[]"))
            # go_aspects MUST be forwarded: omitting it silently scores the union reward, so every
            # aspect_mean run logged rl/reward_fmax and the rollout table against an objective it was
            # not trained on, while rl/reward (from reward_fn) tracked the real one.
            components = [
                rewards.reward_components(
                    text, true_go, obo, ia, reward_weights, go_aspects=go_aspects, go_names=go_names
                )
                for text in texts
            ]
            if os.environ.get("SENPAI_RL_DEBUG") == "1" and (world <= 1 or is_main):
                from bioreason_pro import rewards as _rw
                _obo, _ia = obo, ia
                _w = _rw.RewardWeights()
                print(f"[RLDBG] step {gstep} protein={protein_id} "
                      f"gt_raw={ {k: str(v)[:40] for k, v in gt.items()} } "
                      f"gt_terms(n={len(true_go)})={sorted(true_go)[:8]}", flush=True)
                for j, t in enumerate(texts):
                    print(f"[RLDBG]  rollout{j}: reward={float(r_b[j]):.4f} "
                          f"r_fmax={_rw.r_fmax(t, true_go, _obo, _ia):.4f} "
                          f"r_format={_rw.r_format(t):.3f} r_conc={_rw.r_conciseness(t, _w):.3f} "
                          f"len={len(t)} n_pred_go={len(_rw.extract_go_terms(t))}", flush=True)
                    print(f"[RLDBG]   completion[:500]={t[:500]!r}", flush=True)
            groups.append(dict(protein_id=protein_id, prompt_text=prompt_text, seq=seq,
                                texts=texts, comp_ids=comp_ids, r=r_b, components=components))
        if os.environ.get("SENPAI_RL_DEBUG") == "1" and (world <= 1 or is_main) and gstep >= 2:
            print("[RLDBG] done", flush=True)
            return 0

        # ---- Phase 2: one pooled-std computation across the whole B*G batch (paper Eq. 4.19-4.20).
        # Per-protein numerator is UNCHANGED — each rollout is still centred on its own protein's
        # group mean; only the denominator optionally pools across the batch instead of being 1.0.
        total_n = B * G
        advantages, global_std = _pooled_advantages(
            [g["r"] for g in groups], args.rl_advantage_normalization
        )
        for g, adv in zip(groups, advantages):
            g["adv"] = adv

        # ---- Phase 3: nested backward loop, one fused forward-graph alive at a time (unchanged
        # OOM-avoidance pattern — the existing code already fully materializes all rollouts of a
        # step before any backward() call, so nesting an outer loop over B around the same
        # per-completion body introduces no new memory-pressure pattern).
        opt.zero_grad()
        kl_sum = 0.0
        n_skipped_nonfinite = 0
        for g in groups:
            for cid, a in zip(g["comp_ids"], g["adv"]):
                if cid.shape[1] == 0:
                    continue
                # A degenerate protein group (all G rollouts scored identically — expected and
                # already tolerated, ADR-013) gives every rollout in it advantage exactly 0. At
                # B>1 that 0 can meet an unrelated, real 0-probability token elsewhere in a long
                # (up to max_completion_length) generation and produce 0 * -inf = NaN, which the
                # grad_norm check below would otherwise turn into a full 8-GPU job crash over one
                # bad completion out of B*G. Skip just that completion (counted, not silent) rather
                # than let it poison the whole step's gradient — the aggregate grad_norm check
                # after the loop still fails closed if the run is broken more systemically than this.
                if not torch.isfinite(a):
                    n_skipped_nonfinite += 1
                    continue
                # Backward per completion so only ONE fused forward-graph is alive at a time (grads
                # accumulate in .grad) — avoids holding B*G graphs → the OOM at 16-GPU scale.
                pol = _fused_completion_logps(model, tok, g["prompt_text"], g["seq"], cid, args)
                if not torch.isfinite(pol).all():
                    n_skipped_nonfinite += 1
                    continue
                # per-token MEAN (not sum): keeps updates length-invariant and makes beta interpretable
                # (a sum-over-~512-tokens objective produced oversized updates → the earlier 0.44→0.19
                # drift). Divides by total_n (B*G), not G — at B=1 this is identical to today.
                loss_i = -(a.item() / total_n) * pol.mean()
                if args.beta > 0:  # KL anchor to the frozen SFT reference (adapter disabled) —
                    ref = _fused_completion_logps(model, tok, g["prompt_text"], g["seq"], cid, args,
                                                   ref=True)                          # prevents the
                    if not torch.isfinite(ref).all():
                        n_skipped_nonfinite += 1
                        continue
                    log_ratio = ref - pol                                # policy from drifting away
                    kl = torch.exp(log_ratio) - log_ratio - 1.0          # from the SFT init (k3, >=0)
                    loss_i = loss_i + (args.beta / total_n) * kl.mean()
                    kl_sum += float(kl.mean().detach())
                if getattr(loss_i, "requires_grad", False):
                    loss_i.backward()
        if n_skipped_nonfinite and is_main:
            print(f"MM-GRPO step {gstep}: skipped {n_skipped_nonfinite} non-finite completion(s) "
                  "before backward", flush=True)
        if world > 1:  # manual DDP: average trainable grads across ranks (same param set every step)
            import torch.distributed as dist
            for p in trainable:
                g_ = p.grad if p.grad is not None else torch.zeros_like(p)
                dist.all_reduce(g_, op=dist.ReduceOp.SUM)
                p.grad = g_ / world
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"GRPO step {gstep}: non-finite gradient norm {float(grad_norm)}"
            )
        opt.step()

        # ---- Phase 4: telemetry, aggregated across all B proteins. Every existing key's MEANING is
        # unchanged at B=1 (total_n == G, denom == 1.0 == today's behaviour exactly).
        last_reward = float(torch.cat([g["r"] for g in groups]).mean())
        if is_main:
            kl_mean = kl_sum / total_n
            group_means = [float(g["r"].mean()) for g in groups]
            all_components = [c for g in groups for c in g["components"]]
            rollout_lengths = [int(ids.shape[1]) for g in groups for ids in g["comp_ids"]]
            adv_abs_mean = float(torch.cat([g["adv"] for g in groups]).abs().mean())
            telemetry = {
                "rl/reward": last_reward,
                "rl/reward_std": float(global_std),
                "rl/group_reward_mean": last_reward,
                "rl/group_reward_mean_min": min(group_means),
                "rl/group_reward_mean_max": max(group_means),
                "rl/proteins_per_step": B,
                "rl/skipped_nonfinite_completions": n_skipped_nonfinite,
                # The value actually used as the advantage denominator this step — 0.0 (a stable "off"
                # sentinel, not a hole) when rl_advantage_normalization leaves Dr.GRPO's no-std
                # baseline in place.
                "rl/global_reward_std": (
                    float(global_std + ADV_STD_EPS)
                    if args.rl_advantage_normalization == "group_mean_global_std" else 0.0
                ),
                "rl/advantage_mean_abs": adv_abs_mean,
                "rl/reward_fmax": sum(item.fmax for item in all_components) / total_n,
                "rl/reward_format": sum(item.format for item in all_components) / total_n,
                "rl/reward_conciseness": sum(item.conciseness for item in all_components) / total_n,
                # Logged unconditionally: it is 0.0 under the shipped weights, so the curve shows
                # whether the specificity variant actually changed what the rollouts emit.
                "rl/reward_redundancy": sum(item.redundancy for item in all_components) / total_n,
                # ADR-014 reasoning terms, always measured so an unreasoned arm has a baseline.
                "rl/reward_substance": sum(item.substance for item in all_components) / total_n,
                "rl/reward_faithfulness": sum(item.faithfulness for item in all_components) / total_n,
                "rl/reward_truncated": sum(item.truncation for item in all_components) / total_n,
                # ADR-013: a component with no WITHIN-GROUP variance contributes exactly nothing to a
                # GRPO advantage, because advantages are centred inside the group. Scoped PER PROTEIN
                # and averaged across B (not pooled across proteins) so this keeps measuring
                # within-group variance, not cross-protein variance, at B>1.
                **{
                    f"rl/reward_{name}_sd": sum(
                        _group_sd([getattr(item, name) for item in g["components"]]) for g in groups
                    ) / B
                    for name in ("fmax", "format", "substance", "faithfulness", "truncation")
                },
                "rl/kl": kl_mean,
                "rl/completion_length": sum(rollout_lengths) / total_n,
                "rl/grad_norm": float(grad_norm),
                "rl/step": gstep,
            }
            if wandb_run_id:
                try:
                    import wandb
                    if gstep == 1 or gstep % 10 == 0:
                        telemetry["rl/rollout_samples"] = wandb.Table(
                            columns=["protein_id", "step", "completion", "reward", "fmax"],
                            data=[
                                [g["protein_id"], gstep, text, item.total, item.fmax]
                                for g in groups
                                for text, item in zip(g["texts"], g["components"])
                            ],
                        )
                    wandb.log(telemetry)
                except Exception:
                    pass
            print(f"MM-GRPO step {gstep} reward={last_reward:.4f} std={float(global_std):.4f} "
                  f"kl={kl_mean:.4f}", flush=True)
        stop = 1 if (deadline is not None and time.monotonic() > deadline) else 0
        if world > 1:
            import torch.distributed as dist
            t = torch.tensor([stop], device=dev)
            dist.broadcast(t, src=0)
            stop = int(t.item())
        if stop:
            break

    # Tear down BEFORE the rank-0-only eval (see run_sft: avoids the NCCL barrier-timeout abort).
    _dist_barrier(world)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    if is_main:
        out_dir = args.output_dir or f"outputs/rl-{args.wandb_name or 'run'}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.text_model.save_pretrained(out_dir)
        print(f"MM-GRPO: saved LoRA adapter to {out_dir}", flush=True)
        _save_mm_extras(model, args, out_dir)
        _log_and_emit_eval(model, tok, args, wandb_run_id, multimodal=True)
        _log_checkpoint_artifact(out_dir, args, wandb_run)
        if args.smoke:
            print("MM-GRPO-SMOKE-OK", flush=True)
    return 0


def run_rl(
    args: RunArgs,
    deadline,
    wandb_run_id: str = "",
    ckpt_version: str = "",
    wandb_run=None,
) -> int:
    """GRPO (baseline) / GSPO (opt-in) RL. Text-only path is runnable now (Stage 0/1). The multimodal
    model assembly (ESM2/GO fusion) is wired in build_multimodal_policy(); RL rollout DECODE through
    the fusion is the remaining vendor step (protein-aware vLLM)."""
    from peft import LoraConfig

    import data
    from bioreason_pro import model as M

    if args.use_multimodal:
        # Assembly is wired (build_multimodal_policy → build_model). Opt-in GPU smoke of the assembly
        # + one fwd/bwd through the scatter-add fusion: SENPAI_MM_BUILD=1.
        # (trl is imported lazily in the text branch below so this assembly smoke doesn't require it.)
        if os.environ.get("SENPAI_MM_BUILD") == "1":
            return _multimodal_build_smoke(args, wandb_run_id)
        # Multimodal GRPO: TRL's GRPOTrainer rollout+logprob path can't thread a custom protein modality
        # (it only knows text/images), so use the self-contained DR-GRPO loop over the fused decode.
        return _run_multimodal_grpo(
            args, deadline, wandb_run_id, ckpt_version, wandb_run=wandb_run
        )

    # --- Text-only Stage 0/1 (runnable) ---
    model, tok = M.build_text_model(M.ModelConfig(use_multimodal=False), args.model_name or None)
    reward_fn = _load_reward(args, wandb_run_id, ckpt_version)
    train_dataset = data.tiny_rl_smoke_dataset() if args.smoke else data.load_rl_prompts_text(args.rl_num_prompts)

    from trl import GRPOTrainer

    cfg = build_grpo_config(args)
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=cfg, train_dataset=train_dataset,
                          processing_class=tok, peft_config=peft_cfg)
    trainer.add_callback(_deadline_callback(deadline))
    trainer.train(resume_from_checkpoint=args.resume_from or None)
    trainer.save_model(cfg.output_dir)
    _save_data_manifest(args, cfg.output_dir)

    # Log the RL improvement signal (final mean reward), then score a REAL val_primary/weighted_fmax
    # by generating over the val split (text path supports generation) and emit the SENPAI-RESULT.
    final_reward = _last_logged(trainer, "reward", _last_logged(trainer, "train/reward"))
    if wandb_run_id:
        try:
            import wandb
            wandb.log({"rl/final_reward": final_reward})
        except Exception:
            pass
    _log_and_emit_eval(trainer.model, tok, args, wandb_run_id, multimodal=False)
    return 0


def emit_result(wandb_run_id: str, val_fmax: float, test_fmax: float) -> None:
    """Emit the exact senpai result marker (single line)."""
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [wandb_run_id],
        "primary_metric": {"name": "val_primary/weighted_fmax", "value": val_fmax},
        "test_metric": {"name": "test_primary/weighted_fmax", "value": test_fmax},
    }
    print("SENPAI-RESULT: " + json.dumps(marker, separators=(",", ":")))


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    for mod, fn in (("numpy", "random.seed"), ("torch", "manual_seed")):
        try:
            m = __import__(mod)
            (m.random.seed if mod == "numpy" else m.manual_seed)(seed)
        except Exception:
            pass


def _init_tracking(args: RunArgs):
    """Initialize the rank-0 W&B run used for metrics, Artifacts, and Weave trace identity."""
    import wandb
    import weave

    # Weave's on-disk response cache uses sqlite; SLURM's TMPDIR / NFS $HOME can't open it. Disable
    # it (we only WRITE traces here) and point any residual cache at node-local /tmp.
    os.environ.setdefault("WEAVE_USE_SERVER_CACHE", "false")
    os.environ.setdefault("WEAVE_SERVER_CACHE_DIR", "/tmp/weave_cache")

    # The experiment variants live in the environment, not in RunArgs, so they would otherwise be
    # invisible in W&B — and a condition search whose runs do not record their condition is not
    # auditable. Resolve them through the same fail-closed readers the training path uses.
    from bioreason_pro.data_contract import active_target_variant
    from bioreason_pro.rewards import active_reward_variant
    from bioreason_pro.wandb_meta import build_note, derived_tags
    from eval_targets.base import active_prompt_variant

    job_type = f"{args.stage}-training"
    config = {
        **vars(args),
        "target_variant": active_target_variant(),
        "prompt_variant": active_prompt_variant(),
        # Only on RL: the reward variant has no effect on an SFT run, and recording it there
        # would imply it was used. Its absence made the aspect_mean-versus-union arms
        # indistinguishable in W&B — the condition existed only in the job's stdout.
        **({"reward_variant": active_reward_variant()} if args.stage == "rl" else {}),
    }
    run = wandb.init(
        entity="wandb-healthcare",
        project="bioreasonpro-senpai",
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        job_type=job_type,
        config=config,
        tags=derived_tags(job_type, config),
        notes=build_note(job_type, config),
    )
    weave.init("wandb-healthcare/bioreasonpro-senpai")
    return run


def main() -> int:
    args = simple_parsing.parse(RunArgs)
    args.validate()
    _seed_everything(args.resolved_train_seed())
    deadline = args.timeout_deadline()
    # Only rank 0 owns W&B/Weave (avoid 16 duplicate runs under torchrun); other ranks get "".
    run = _init_tracking(args) if int(os.environ.get("RANK", "0")) == 0 else None
    run_id = run.id if run else ""
    _resolve_sft_artifact(args, run)
    if run:
        run.config.update(vars(args), allow_val_change=True)
    if args.stage == "sft":
        result = run_sft(
            args,
            deadline,
            wandb_run_id=run_id,
            ckpt_version=args.resume_from or "base",
            wandb_run=run,
        )
    else:
        result = run_rl(
            args,
            deadline,
            wandb_run_id=run_id,
            ckpt_version=args.resume_from or "base",
            wandb_run=run,
        )
    if run_id:
        import wandb

        wandb.finish()
    return result


if __name__ == "__main__":
    sys.exit(main())
