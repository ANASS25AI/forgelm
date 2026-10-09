import json
from pathlib import Path

import pytest

from forgelm.tokenization import CharacterTokenizer


def test_train_encode_decode_round_trip() -> None:
    text = "hello\n"
    tokenizer = CharacterTokenizer.train(text)

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert tokenizer.vocab_size == len(set(text))


def test_vocabulary_is_deterministic() -> None:
    first = CharacterTokenizer.train("cab")
    second = CharacterTokenizer.train("bca")

    assert first.vocabulary == second.vocabulary == ("a", "b", "c")


def test_unknown_character_is_rejected() -> None:
    tokenizer = CharacterTokenizer.train("abc")

    with pytest.raises(ValueError, match="unknown characters"):
        tokenizer.encode("abcd")


def test_tokenizer_serialization(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.train("hello\n")
    output_path = tmp_path / "tokenizer.json"

    tokenizer.save(output_path)
    loaded = CharacterTokenizer.load(output_path)

    assert loaded == tokenizer
    assert json.loads(output_path.read_text(encoding="utf-8"))["type"] == "character"

