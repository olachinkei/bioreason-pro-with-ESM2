"""Regression checks for the SFT CUDA device contract."""

from pathlib import Path


TRAIN_SOURCE = (Path(__file__).resolve().parents[1] / "train.py").read_text(encoding="utf-8")


def test_sft_cuda_memory_apis_use_the_current_device():
    assert 'cuda_device = torch.device("cuda", local)' in TRAIN_SOURCE
    for api in (
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
    ):
        assert f"torch.cuda.{api}()" in TRAIN_SOURCE
        assert f"torch.cuda.{api}(local)" not in TRAIN_SOURCE
        assert f"torch.cuda.{api}(cuda_device)" not in TRAIN_SOURCE
