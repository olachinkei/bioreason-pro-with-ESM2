"""ESM2 protein encoder for the commercially permissive workflow.

The only supported protein checkpoint is ``facebook/esm2_t33_650M_UR50D`` (MIT), loaded through
``transformers.EsmModel``. Other model names fail the repository allowlist before transformers or
model weights are loaded.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from bioreason_pro.license_policy import approved_model_revision, require_approved_model

MAX_LENGTH_PROTEIN = 1024


class ProteinEncoder(ABC):
    """Abstract interface consumed by the multimodal fusion model."""

    def __init__(self, model_name: str, inference_mode: bool = True):
        self.model_name = model_name
        self.model = None
        self._embedding_dim = None
        self.inference_mode = inference_mode
        self._protein_train_layer_start = 36

    @abstractmethod
    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return one concatenated per-token embedding tensor per batch item."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the encoder hidden size."""

    @property
    @abstractmethod
    def supports_structure(self) -> bool:
        """Whether structure coordinates are accepted."""

    @abstractmethod
    def setup_training(self, protein_train_layer_start: int = 36):
        """Configure which encoder layers, if any, are trainable."""

    def set_inference_mode(self, inference_mode: bool, protein_train_layer_start: int = 36):
        self.inference_mode = inference_mode
        self._protein_train_layer_start = protein_train_layer_start
        if inference_mode:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.setup_training(protein_train_layer_start)


class ESM2Encoder(ProteinEncoder):
    """Hugging Face ESM2 encoder returning layer-selected token embeddings.

    ESM2 tokenization adds BOS/EOS, so an N-residue input produces N+2 rows. This is the shape
    contract used by ``data.expand_pad_tokens`` and the fusion layer.
    """

    def __init__(
        self,
        model_name: str,
        inference_mode: bool = True,
        embedding_layer: int = 33,
        dtype: Optional[torch.dtype] = None,
    ):
        approved_name = require_approved_model(model_name, "protein")
        super().__init__(approved_name, inference_mode)

        from transformers import AutoTokenizer, EsmModel

        # bf16 on GPU, fp32 on CPU. The encoder is frozen in the default configuration and the
        # fusion projection casts to its own dtype anyway, so bf16 buys memory and bandwidth
        # without changing what the projection sees; CPU smokes stay fp32 so they remain exact.
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.dtype = dtype

        revision = approved_model_revision(approved_name, "protein")
        self.tokenizer = AutoTokenizer.from_pretrained(approved_name, revision=revision)
        # add_pooling_layer=False: the fusion path consumes per-residue hidden states only. The
        # pooler is never called, but it would still be materialised and counted as untrained
        # parameters in the trainable-parameter accounting.
        self.model = EsmModel.from_pretrained(
            approved_name, revision=revision, add_pooling_layer=False, dtype=dtype
        )
        self._embedding_dim = int(self.model.config.hidden_size)
        self.total_blocks = int(self.model.config.num_hidden_layers)
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.max_residues = int(self.model.config.max_position_embeddings) - special_tokens
        if self.max_residues != MAX_LENGTH_PROTEIN:
            raise ValueError(
                f"{approved_name}: config/tokenizer allow {self.max_residues} residues "
                f"({self.model.config.max_position_embeddings} positions - {special_tokens} "
                f"special tokens), but the fusion contract is {MAX_LENGTH_PROTEIN}"
            )
        self.embedding_layer = self._validate_embedding_layer(embedding_layer)
        self.set_inference_mode(self.inference_mode)

    def _validate_embedding_layer(self, embedding_layer: int) -> int:
        """Validate an index into HF's hidden_states tuple (0=embeddings, N=layer N)."""
        if 0 <= embedding_layer <= self.total_blocks:
            return embedding_layer
        raise ValueError(
            f"embedding_layer must be 0-{self.total_blocks} for {self.model_name}, "
            f"got {embedding_layer}"
        )

    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode sequences with ESM2, preserving BOS/EOS and the fusion token cap."""
        del structure_coords
        result = [[] for _ in range(batch_size)]
        device = next(self.model.parameters()).device

        for seq_idx, sequence in enumerate(protein_sequences):
            sequence = sequence[: self.max_residues]
            encoded = self.tokenizer(
                sequence,
                add_special_tokens=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.set_grad_enabled(not self.inference_mode):
                output = self.model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
                seq_embeddings = output.hidden_states[self.embedding_layer].squeeze(0)

            expected_tokens = len(sequence) + self.tokenizer.num_special_tokens_to_add(pair=False)
            if seq_embeddings.shape[0] != expected_tokens:
                raise ValueError(
                    f"ESM2 produced {seq_embeddings.shape[0]} tokens for {len(sequence)} residues; "
                    f"expected {expected_tokens} including BOS/EOS"
                )
            result[batch_idx_map[seq_idx]].append(seq_embeddings)

        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                # Match the populated branch's dtype so a protein-free batch item does not hand the
                # fusion layer an fp32 tensor while every other item is bf16.
                result[i] = torch.zeros(
                    (0, self.embedding_dim), device=device, dtype=self.dtype
                )
        return result

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def supports_structure(self) -> bool:
        return False

    def setup_training(self, protein_train_layer_start: int = 36):
        """Freeze ESM2 first, then optionally unfreeze layers from the requested index."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        if protein_train_layer_start == -1 or protein_train_layer_start >= self.total_blocks:
            print(
                "✓ ESM2 protein encoder training setup: "
                f"0/{self.total_blocks} transformer layers trainable"
            )
            return

        start_layer = max(0, protein_train_layer_start)
        for layer in self.model.encoder.layer[start_layer:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True
        print(
            "✓ ESM2 protein encoder training setup: "
            f"{self.total_blocks - start_layer}/{self.total_blocks} transformer layers trainable"
        )


def create_protein_encoder(
    model_name: str,
    inference_mode: bool = True,
    embedding_layer: int = 33,
    dtype: Optional[torch.dtype] = None,
) -> ProteinEncoder:
    """Create the single supported ESM2 encoder after an exact allowlist check.

    ``dtype=None`` selects bf16 on GPU and fp32 on CPU.
    """
    approved_name = require_approved_model(model_name, "protein")
    return ESM2Encoder(approved_name, inference_mode, embedding_layer, dtype)
