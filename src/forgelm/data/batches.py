"""Create next-token-prediction datasets and mini-batches."""

from __future__ import annotations

import torch
from torch import Tensor


def split_tokens(tokens: Tensor, train_fraction: float = 0.9) -> tuple[Tensor, Tensor]:
    """Split one token sequence into non-overlapping training and validation sets."""

    if tokens.ndim != 1:
        raise ValueError("Expected a one-dimensional token sequence.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    split_index = int(len(tokens) * train_fraction)
    if split_index < 2 or len(tokens) - split_index < 2:
        raise ValueError("Both splits need at least two tokens.")
    return tokens[:split_index], tokens[split_index:]


def get_batch(
    tokens: Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample context windows and their one-token-shifted targets.

    Both returned tensors have shape ``[batch_size, block_size]``.
    """

    if tokens.ndim != 1:
        raise ValueError("Expected a one-dimensional token sequence.")
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive.")
    if len(tokens) <= block_size:
        raise ValueError("Token sequence must be longer than block_size.")

    starts = torch.randint(
        low=0,
        high=len(tokens) - block_size,
        size=(batch_size,),
        generator=generator,
    )
    inputs = torch.stack([tokens[start : start + block_size] for start in starts.tolist()])
    targets = torch.stack(
        [tokens[start + 1 : start + block_size + 1] for start in starts.tolist()]
    )
    return inputs.to(device), targets.to(device)

