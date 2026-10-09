"""The smallest trainable autoregressive language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        self.vocab_size = vocab_size
        self.transition_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return vocabulary logits and optional mean cross-entropy loss.

        ``token_ids`` has shape ``[B, T]`` and the returned logits have shape
        ``[B, T, V]``.
        """

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time].")

        logits = self.transition_logits(token_ids)
        loss: Tensor | None = None

        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids.")
            batch_size, sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively sample new tokens from the model."""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if top_k is not None and not 0 < top_k <= self.vocab_size:
            raise ValueError("top_k must be between 1 and vocab_size.")

        generated = token_ids
        for _ in range(max_new_tokens):
            logits, _ = self(generated[:, -1:])
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                threshold = torch.topk(next_logits, top_k).values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)

        return generated

