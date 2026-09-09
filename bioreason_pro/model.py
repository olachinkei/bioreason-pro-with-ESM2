"""ProteinLLMModel — Qwen3-4B-Thinking + ESM2 + optional GO-GAT fusion (PROTECTED).

Vendored/adapted from public bioreason2/models/protein_llm.py. The pad-token scatter-add FUSION
(the correctness-critical, subtle part) is factored into `scatter_modality_embeds` and unit-tested
on tiny CPU tensors. The full model ASSEMBLY (ESM2 encoder, GO-GAT, projections, LLM) needs
GPU/transformers/torch-geometric and is laid out in build_model as the vendor/GPU TODO.

Supported model constants:
  Qwen3-4B-Thinking-2507: hidden_size=2560. ESM2-650M: d_model=1280, layer 33.
  Projection input size is still read from the loaded approved encoder at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

QWEN_HIDDEN = 2560          # Qwen3-4B-Thinking-2507 hidden_size
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_D_MODEL = 1280
ESM2_LAYER = 33
GO_NODE_DIM = 2560          # GO node / embedding dim


@dataclass
class ModelConfig:
    text_model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    esm_model_name: str = ESM2_MODEL_NAME
    esm_layer: int = ESM2_LAYER
    max_length_protein: int = 1024
    max_length_text: int = 10000
    use_multimodal: bool = True
    freeze_esm: bool = True
    # LoRA — SFT r=128/α=256; RL r=16/α=32 (set per stage by train.py). dropout 0.05.
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    # GO-GAT (upstream arg names: go_graph_encoder.create_go_graph_encoder_pipeline)
    go_obo_path: str = "data/go-basic.obo"
    go_embeddings_path: str = ""          # DIRECTORY of GO_*.safetensors precomputed node embeddings
    go_hidden_dim: int = 512
    go_num_gat_layers: int = 3
    go_num_heads: int = 8
    go_num_reduced_embeddings: int = 200  # the fixed 200 GO tokens per aspect (matches data pad count)
    go_embedding_dim: int = GO_NODE_DIM
    unified_go_encoder: bool = True


def scatter_modality_embeds(text_inputs_embeds: torch.Tensor, pad_mask: torch.Tensor,
                            modality_embeds_flat: torch.Tensor) -> torch.Tensor:
    """Replace embeddings at `pad_mask` positions with `modality_embeds_flat`, autograd-safe.

    Verbatim mechanism from bioreason2 protein_llm.forward: flatten (B,L,H)->(B*L,H), compute the
    (modality - existing) diff at the flattened pad indices, and scatter_add it OUT-OF-PLACE
    (mirrors masked_scatter without an in-place write, so the autograd graph / vLLM graph replay
    stays intact). Raises if #pad positions != #modality rows — that count is the shape contract
    with the encoder output length (protein: min(len(seq),1024)+2; go: 200).
    """
    orig_shape = text_inputs_embeds.shape            # (B, L, H)
    hidden_size = orig_shape[-1]
    embeds_2d = text_inputs_embeds.view(-1, hidden_size)
    idx = pad_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
    n_tokens = int(idx.shape[0])
    n_features = int(modality_embeds_flat.shape[0])
    if n_features != n_tokens:
        raise ValueError(f"modality features ({n_features}) != pad tokens ({n_tokens})")
    if n_tokens == 0:
        return text_inputs_embeds
    modality = modality_embeds_flat.to(dtype=embeds_2d.dtype)
    diff = modality - embeds_2d.index_select(0, idx)
    embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
    return embeds_2d.view(orig_shape)


def fuse_embeddings(text_inputs_embeds: torch.Tensor, input_ids: torch.Tensor,
                    protein_token_id: int, go_token_id: int,
                    protein_embeds_flat: torch.Tensor | None = None,
                    go_embeds_flat: torch.Tensor | None = None) -> torch.Tensor:
    """Inject already-projected protein then GO embeddings at their pad-token positions (train path).

    `*_embeds_flat` are the projection outputs (protein_projection / go_projection → text hidden),
    concatenated across the batch in batch_idx_map order. Mirrors protein_llm.forward exactly.
    """
    if protein_embeds_flat is not None:
        text_inputs_embeds = scatter_modality_embeds(
            text_inputs_embeds, input_ids == protein_token_id, protein_embeds_flat)
    if go_embeds_flat is not None:
        text_inputs_embeds = scatter_modality_embeds(
            text_inputs_embeds, input_ids == go_token_id, go_embeds_flat)
    return text_inputs_embeds


def build_text_model(cfg: ModelConfig, model_name: str | None = None):
    """Text-only assembly (Stage 0/1, no ESM/GAT): AutoModelForCausalLM + tokenizer with the two
    pad tokens added + embeddings resized. Runnable now (no protein/GO encoders). Returns (model, tok).
    """
    from bioreason_pro.license_policy import approved_model_revision, require_approved_model

    from bioreason_pro.special_tokens import ALL_SPECIAL_TOKENS

    name = require_approved_model(model_name or cfg.text_model_name, "text")
    revision = approved_model_revision(name, "text")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype="auto")
    tok.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    model.resize_token_embeddings(len(tok))
    return model, tok


class ProteinLLMModel(nn.Module):
    """Qwen3 + ESM2 + (optional) GO-GAT multimodal fusion.

    Composes: a causal LM (Qwen3, LoRA-trained by train.py), a frozen ESM2 protein encoder,
    an optional GO-GAT encoder, and two MLP projections into the
    LM hidden size. `forward` embeds the text, replaces `<|protein_pad|>` / `<|go_graph_pad|>`
    positions with projected modality embeddings via the autograd-safe scatter_add in
    `fuse_embeddings`, then runs the LM.
    """

    def __init__(self, text_model, text_tokenizer, protein_encoder, go_encoder,
                 protein_projection, go_projection, protein_token_id, go_token_id,
                 text_hidden_size, unified_go_encoder):
        super().__init__()
        self.text_model = text_model                      # nn.Module (LoRA target)
        self.protein_model = protein_encoder.model        # ESM2 nn.Module; registered for .to()
        self.protein_projection = protein_projection      # nn.Module (trained)
        self.go_encoder = go_encoder                      # nn.Module or None
        self.go_projection = go_projection                # nn.Module or None
        self.protein_encoder = protein_encoder            # plain wrapper (holds .model, .encode_sequences)
        self.text_tokenizer = text_tokenizer
        self.protein_token_id = protein_token_id
        self.go_token_id = go_token_id
        self.text_hidden_size = text_hidden_size
        self.unified_go_encoder = unified_go_encoder

    def _process_protein(self, protein_sequences, batch_idx_map, batch_size, structure_coords=None):
        """ESM2-encode each sequence and project to the LM hidden size."""
        proj = self.protein_projection
        embeds = self.protein_encoder.encode_sequences(
            protein_sequences=protein_sequences, batch_idx_map=batch_idx_map,
            batch_size=batch_size, structure_coords=structure_coords)
        for i in range(batch_size):
            if embeds[i].numel() > 0:
                embeds[i] = proj(embeds[i].to(device=proj[0].weight.device, dtype=proj[0].weight.dtype))
            else:
                embeds[i] = torch.zeros((0, self.text_hidden_size),
                                        device=proj[0].weight.device, dtype=proj[0].weight.dtype)
        return embeds

    def _process_go(self, batch_size):
        """Produce the (200, hidden) GO-GAT tokens and replicate per batch item."""
        if self.go_encoder is None:
            return None
        reduced = self.go_encoder("all")  # (200, 2560)
        if self.go_projection is not None:
            gp = self.go_projection
            reduced = gp(reduced.to(device=gp[0].weight.device, dtype=gp[0].weight.dtype))
        return [reduced for _ in range(batch_size)]

    def forward(self, input_ids=None, attention_mask=None, protein_sequences=None,
                batch_idx_map=None, structure_coords=None, labels=None, go_aspects=None, **kwargs):
        """Embed text, fuse projected protein/GO embeddings at their pad positions, run the LM."""
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")
        batch_size = input_ids.shape[0]
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        protein_flat = None
        if protein_sequences is not None and batch_idx_map is not None:
            protein_flat = torch.cat(
                self._process_protein(protein_sequences, batch_idx_map, batch_size, structure_coords), dim=0)

        go_flat = None
        if go_aspects is not None and self.go_encoder is not None:
            go_list = self._process_go(batch_size)
            go_flat = torch.cat([e for e in go_list if e.numel() > 0], dim=0) if go_list else None

        text_inputs_embeds = fuse_embeddings(
            text_inputs_embeds, input_ids, self.protein_token_id, self.go_token_id,
            protein_embeds_flat=protein_flat, go_embeds_flat=go_flat)

        return self.text_model(inputs_embeds=text_inputs_embeds, attention_mask=attention_mask,
                               labels=labels, **kwargs)


def build_model(cfg: ModelConfig, model_name: str | None = None):
    """Assemble Qwen3 + a protein encoder (+ optional GO-GAT) into a fused model.

    Needs the train env (transformers and — if cfg.go_embeddings_path is set — torch-geometric)
    and a GPU. The protein encoder is frozen when cfg.freeze_esm is true. Its reported embedding
    dimension drives the projection input (ESM2-650M=1280).
    The GO encoder is built only when cfg.go_embeddings_path is provided (else protein-only fusion).
    LoRA is applied by train.py (peft) on the returned model.
    """
    from bioreason_pro.license_policy import (
        approved_model_revision,
        require_approved_model,
        require_no_unapproved_go_assets,
    )

    name = require_approved_model(model_name or cfg.text_model_name, "text")
    text_revision = approved_model_revision(name, "text")
    require_approved_model(cfg.esm_model_name, "protein")
    require_no_unapproved_go_assets(cfg.go_embeddings_path)

    from bioreason_pro.protein_encoder import create_protein_encoder
    from bioreason_pro.special_tokens import ALL_SPECIAL_TOKENS, GO_GRAPH_PAD_TOKEN, PROTEIN_PAD_TOKEN

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=text_revision)
    try:
        text_model = AutoModelForCausalLM.from_pretrained(
            name, revision=text_revision, dtype=torch.bfloat16,
            attn_implementation="flash_attention_2")
    except Exception:
        text_model = AutoModelForCausalLM.from_pretrained(
            name, revision=text_revision, dtype=torch.bfloat16)
    tok.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    text_model.resize_token_embeddings(len(tok))
    protein_token_id = tok.convert_tokens_to_ids(PROTEIN_PAD_TOKEN)
    go_token_id = tok.convert_tokens_to_ids(GO_GRAPH_PAD_TOKEN)
    text_hidden = text_model.config.hidden_size

    protein_encoder = create_protein_encoder(
        cfg.esm_model_name, inference_mode=cfg.freeze_esm, embedding_layer=cfg.esm_layer)
    protein_hidden = protein_encoder.embedding_dim

    go_encoder, go_projection = None, None
    if cfg.go_embeddings_path:
        from bioreason_pro.go_graph_encoder import create_go_graph_encoder_pipeline
        go_encoder = create_go_graph_encoder_pipeline(
            go_obo_path=cfg.go_obo_path, precomputed_embeddings_path=cfg.go_embeddings_path,
            hidden_dim=cfg.go_hidden_dim, num_gat_layers=cfg.go_num_gat_layers,
            num_heads=cfg.go_num_heads, num_reduced_embeddings=cfg.go_num_reduced_embeddings,
            embedding_dim=cfg.go_embedding_dim, unified_go_encoder=cfg.unified_go_encoder)
        go_projection = nn.Sequential(
            nn.Linear(cfg.go_embedding_dim, text_hidden), nn.GELU(),
            nn.Linear(text_hidden, text_hidden)).to(text_model.device, dtype=torch.bfloat16)

    protein_projection = nn.Sequential(
        nn.Linear(protein_hidden, text_hidden), nn.GELU(),
        nn.Linear(text_hidden, text_hidden)).to(text_model.device, dtype=torch.bfloat16)

    return ProteinLLMModel(
        text_model=text_model, text_tokenizer=tok, protein_encoder=protein_encoder,
        go_encoder=go_encoder, protein_projection=protein_projection, go_projection=go_projection,
        protein_token_id=protein_token_id, go_token_id=go_token_id,
        text_hidden_size=text_hidden, unified_go_encoder=cfg.unified_go_encoder)
