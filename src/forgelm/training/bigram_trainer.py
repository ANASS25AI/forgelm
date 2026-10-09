"""Training utilities for the Stage 1 bigram language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forgelm.data import get_batch
from forgelm.models import BigramLanguageModel


@dataclass(frozen=True)
class BigramTrainingConfig:
    batch_size: int = 32
    block_size: int = 8
    max_steps: int = 500
    eval_interval: int = 100
    eval_batches: int = 20
    learning_rate: float = 1e-2
    seed: int = 1337

    def __post_init__(self) -> None:
        positive_integers = {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_tokens: Tensor,
    validation_tokens: Tensor,
    *,
    config: BigramTrainingConfig,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    """Estimate mean train and validation loss over fresh random batches."""

    model.eval()
    results: dict[str, float] = {}

    for split_name, split_tokens in (
        ("train", train_tokens),
        ("validation", validation_tokens),
    ):
        losses = torch.empty(config.eval_batches)
        for batch_index in range(config.eval_batches):
            inputs, targets = get_batch(
                split_tokens,
                batch_size=config.batch_size,
                block_size=config.block_size,
                device=device,
                generator=generator,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            losses[batch_index] = loss.detach().cpu()
        results[split_name] = losses.mean().item()

    model.train()
    return results


def train_bigram(
    train_tokens: Tensor,
    validation_tokens: Tensor,
    *,
    vocab_size: int,
    config: BigramTrainingConfig,
    device: torch.device,
) -> tuple[BigramLanguageModel, list[dict[str, float | int]]]:
    """Train a bigram model and return the model plus evaluation history."""

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for step in range(config.max_steps + 1):
        should_evaluate = step % config.eval_interval == 0 or step == config.max_steps
        if should_evaluate:
            losses = estimate_loss(
                model,
                train_tokens,
                validation_tokens,
                config=config,
                device=device,
                generator=generator,
            )
            history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "validation_loss": losses["validation"],
                }
            )

        if step == config.max_steps:
            break

        inputs, targets = get_batch(
            train_tokens,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, history

