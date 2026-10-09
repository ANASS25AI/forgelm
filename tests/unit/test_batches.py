import torch

from forgelm.data import get_batch, split_tokens


def test_split_tokens_is_non_overlapping() -> None:
    tokens = torch.arange(20)

    train, validation = split_tokens(tokens, train_fraction=0.75)

    assert train.tolist() == list(range(15))
    assert validation.tolist() == list(range(15, 20))


def test_targets_are_inputs_shifted_by_one_token() -> None:
    tokens = torch.arange(100)

    inputs, targets = get_batch(
        tokens,
        batch_size=4,
        block_size=8,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert torch.equal(inputs + 1, targets)

