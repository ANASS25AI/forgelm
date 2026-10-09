import pytest
import torch

from forgelm.models import BigramLanguageModel


def test_forward_shapes_and_loss() -> None:
    model = BigramLanguageModel(vocab_size=11)
    inputs = torch.randint(0, 11, (4, 8))
    targets = torch.randint(0, 11, (4, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (4, 8, 11)
    assert loss is not None
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_generation_adds_requested_number_of_tokens() -> None:
    model = BigramLanguageModel(vocab_size=5)
    prompt = torch.tensor([[0, 1]])

    generated = model.generate(
        prompt,
        max_new_tokens=7,
        generator=torch.Generator().manual_seed(11),
    )

    assert generated.shape == (1, 9)
    assert torch.equal(generated[:, :2], prompt)


def test_invalid_temperature_is_rejected() -> None:
    model = BigramLanguageModel(vocab_size=5)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.tensor([[0]]), max_new_tokens=1, temperature=0.0)

