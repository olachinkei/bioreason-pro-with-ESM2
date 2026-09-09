"""Pre-GPU units for the multimodal scatter-add fusion (bioreason_pro/model.py).

torch (CPU, dev dependency) only — no ESM/transformers/GPU. This covers the correctness-critical
piece: pad positions replaced, autograd preserved, count contract enforced.
"""

import pytest
import torch

from bioreason_pro import model as M


def test_scatter_replaces_only_masked_positions():
    embeds = torch.zeros(1, 4, 3)                       # (B=1, L=4, H=3)
    mask = torch.tensor([[False, True, True, False]])
    modality = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    out = M.scatter_modality_embeds(embeds, mask, modality)
    assert torch.equal(out[0, 1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(out[0, 2], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.equal(out[0, 0], torch.zeros(3)) and torch.equal(out[0, 3], torch.zeros(3))


def test_scatter_matches_masked_scatter_reference():
    embeds = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1] = mask[0, 3] = mask[1, 0] = True          # 3 masked positions (flattened order)
    modality = torch.randn(3, 4)
    out = M.scatter_modality_embeds(embeds, mask, modality)
    ref = embeds.clone()
    ref[mask] = modality                                 # in-place reference (masked_scatter equivalent)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_scatter_preserves_autograd():
    embeds = torch.zeros(1, 3, 2)
    mask = torch.tensor([[False, True, False]])
    modality = torch.randn(1, 2, requires_grad=True)
    out = M.scatter_modality_embeds(embeds, mask, modality)
    out.sum().backward()
    assert modality.grad is not None and torch.count_nonzero(modality.grad) > 0


def test_scatter_count_mismatch_raises():
    embeds = torch.zeros(1, 3, 2)
    mask = torch.tensor([[True, True, False]])           # 2 positions
    with pytest.raises(ValueError):
        M.scatter_modality_embeds(embeds, mask, torch.randn(1, 2))  # only 1 modality row


def test_scatter_empty_mask_is_noop():
    embeds = torch.randn(1, 3, 2)
    out = M.scatter_modality_embeds(embeds, torch.zeros(1, 3, dtype=torch.bool), torch.empty(0, 2))
    assert torch.equal(out, embeds)


def test_fuse_embeddings_both_modalities():
    embeds = torch.zeros(1, 5, 3)
    input_ids = torch.tensor([[7, 100, 100, 200, 9]])    # protein_id=100 (x2), go_id=200 (x1)
    prot = torch.tensor([[1.0, 1, 1], [1, 1, 1]])
    go = torch.tensor([[5.0, 5, 5]])
    out = M.fuse_embeddings(embeds, input_ids, protein_token_id=100, go_token_id=200,
                            protein_embeds_flat=prot, go_embeds_flat=go)
    assert torch.equal(out[0, 1], torch.ones(3)) and torch.equal(out[0, 2], torch.ones(3))
    assert torch.equal(out[0, 3], torch.tensor([5.0, 5, 5]))
    assert torch.equal(out[0, 0], torch.zeros(3))        # non-pad untouched
