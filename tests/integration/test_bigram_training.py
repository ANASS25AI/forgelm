import torch

from forgelm.data import split_tokens
from forgelm.training import BigramTrainingConfig, train_bigram


def test_training_reduces_loss_on_predictable_sequence() -> None:
    tokens = torch.tensor(([0, 1] * 300), dtype=torch.long)
    train_tokens, validation_tokens = split_tokens(tokens)
    config = BigramTrainingConfig(
        batch_size=16,
        block_size=8,
        max_steps=100,
        eval_interval=100,
        eval_batches=5,
        learning_rate=0.1,
        seed=17,
    )

    _, history = train_bigram(
        train_tokens,
        validation_tokens,
        vocab_size=2,
        config=config,
        device=torch.device("cpu"),
    )

    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]

