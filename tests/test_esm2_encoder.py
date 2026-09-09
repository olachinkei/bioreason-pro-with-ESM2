"""CPU-only contract tests for the ESM2 protein-encoder path."""

from types import ModuleType, SimpleNamespace
import importlib
import sys

import pytest
import torch
from torch import nn


class _FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 2

    def __call__(self, sequence, **kwargs):
        del kwargs
        # BOS + one token per residue + EOS
        return {
            "input_ids": torch.arange(len(sequence) + 2).unsqueeze(0),
            "attention_mask": torch.ones(1, len(sequence) + 2, dtype=torch.long),
        }


class _FakeEsmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            hidden_size=1280, num_hidden_layers=33, max_position_embeddings=1026
        )
        self.encoder = SimpleNamespace(layer=nn.ModuleList([nn.Linear(1, 1) for _ in range(33)]))

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict):
        del attention_mask, output_hidden_states, return_dict
        shape = (1, input_ids.shape[1], self.config.hidden_size)
        hidden_states = tuple((self.anchor * i).expand(shape) for i in range(34))
        return SimpleNamespace(hidden_states=hidden_states)


@pytest.fixture
def protein_encoder_module(monkeypatch):
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda _name, **_kwargs: _FakeTokenizer()
    )
    transformers.EsmModel = SimpleNamespace(
        from_pretrained=lambda _name, **_kwargs: _FakeEsmModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    sys.modules.pop("bioreason_pro.protein_encoder", None)
    yield importlib.import_module("bioreason_pro.protein_encoder")
    sys.modules.pop("bioreason_pro.protein_encoder", None)


def test_factory_routes_esm2_and_selects_final_layer(protein_encoder_module):
    module = protein_encoder_module
    encoder = module.create_protein_encoder(
        "facebook/esm2_t33_650M_UR50D", inference_mode=True, embedding_layer=33
    )
    encoded = encoder.encode_sequences(["MKT", "AA"], [0, 1], batch_size=2)

    assert isinstance(encoder, module.ESM2Encoder)
    assert encoder.embedding_dim == 1280
    assert encoded[0].shape == (5, 1280)
    assert encoded[1].shape == (4, 1280)
    assert torch.all(encoded[0] == 33)
    assert all(not param.requires_grad for param in encoder.model.parameters())


def test_esm2_preserves_1024_residue_plus_special_token_contract(protein_encoder_module):
    encoder = protein_encoder_module.ESM2Encoder(
        "facebook/esm2_t33_650M_UR50D", embedding_layer=33
    )
    encoded = encoder.encode_sequences(["M" * 2500], [0], batch_size=1)
    assert encoded[0].shape == (1026, 1280)
    assert encoder.max_residues == 1024


def test_esm2_load_skips_the_pooler(protein_encoder_module, monkeypatch):
    """The fusion path reads per-residue states only; the pooler would be untrained dead weight."""
    captured = {}

    def _capture(_name, **kwargs):
        captured.update(kwargs)
        return _FakeEsmModel()

    monkeypatch.setattr(sys.modules["transformers"].EsmModel, "from_pretrained", _capture)
    protein_encoder_module.ESM2Encoder("facebook/esm2_t33_650M_UR50D", embedding_layer=33)

    assert captured["add_pooling_layer"] is False
    # The allowlist revision pin must survive alongside the new kwargs.
    assert captured["revision"]


@pytest.mark.parametrize(
    "cuda_available, expected_dtype",
    [(True, torch.bfloat16), (False, torch.float32)],
)
def test_esm2_dtype_defaults_to_bf16_on_gpu_and_fp32_on_cpu(
    protein_encoder_module, monkeypatch, cuda_available, expected_dtype
):
    captured = {}

    def _capture(_name, **kwargs):
        captured.update(kwargs)
        return _FakeEsmModel()

    monkeypatch.setattr(sys.modules["transformers"].EsmModel, "from_pretrained", _capture)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    encoder = protein_encoder_module.ESM2Encoder(
        "facebook/esm2_t33_650M_UR50D", embedding_layer=33
    )

    assert captured["dtype"] is expected_dtype
    assert encoder.dtype is expected_dtype


def test_esm2_explicit_dtype_overrides_the_device_default(protein_encoder_module, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    encoder = protein_encoder_module.ESM2Encoder(
        "facebook/esm2_t33_650M_UR50D", embedding_layer=33, dtype=torch.bfloat16
    )
    assert encoder.dtype is torch.bfloat16


def test_protein_free_batch_item_matches_the_encoder_dtype(protein_encoder_module):
    """An empty item must not hand the fusion layer fp32 while every other item is bf16."""
    encoder = protein_encoder_module.ESM2Encoder(
        "facebook/esm2_t33_650M_UR50D", embedding_layer=33, dtype=torch.bfloat16
    )
    encoded = encoder.encode_sequences(["MKT"], [0], batch_size=2)

    assert encoded[1].shape == (0, 1280)
    assert encoded[1].dtype is torch.bfloat16


def test_esm2_rejects_layer_outside_hidden_state_tuple(protein_encoder_module):
    with pytest.raises(ValueError, match="0-33"):
        protein_encoder_module.ESM2Encoder(
            "facebook/esm2_t33_650M_UR50D", embedding_layer=34
        )


@pytest.mark.parametrize(
    "model_name",
    ["esm3_sm_open_v1", "esmc_600m", "facebook/esm2_t30_150M_UR50D", "unknown/model"],
)
def test_factory_rejects_every_non_allowlisted_encoder_before_loading(
    protein_encoder_module, monkeypatch, model_name
):
    loaded = False

    def _unexpected_load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("model loading must not be reached")

    monkeypatch.setattr(
        sys.modules["transformers"].EsmModel, "from_pretrained", _unexpected_load
    )
    with pytest.raises(ValueError, match="not approved"):
        protein_encoder_module.create_protein_encoder(model_name)
    assert loaded is False


def test_new_training_defaults_use_esm2():
    from bioreason_pro.model import ESM2_MODEL_NAME, ModelConfig
    from train import RunArgs

    assert ModelConfig().esm_model_name == ESM2_MODEL_NAME
    assert ModelConfig().esm_layer == 33
    assert RunArgs().esm_model_name == ESM2_MODEL_NAME
    assert RunArgs().esm_layer == 33
