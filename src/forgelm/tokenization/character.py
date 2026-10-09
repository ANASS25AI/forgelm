"""A minimal character-level tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CharacterTokenizer:
    """Map every unique character to one integer token ID."""

    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError("The vocabulary cannot be empty.")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Every vocabulary character must be unique.")

    @classmethod
    def train(cls, text: str) -> CharacterTokenizer:
        """Construct a deterministic vocabulary from the characters in ``text``."""

        if not text:
            raise ValueError("Cannot train a tokenizer on empty text.")
        return cls(vocabulary=tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocabulary)}

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, rejecting characters outside the vocabulary."""

        mapping = self.token_to_id
        unknown = sorted(set(text) - mapping.keys())
        if unknown:
            formatted = ", ".join(repr(character) for character in unknown)
            raise ValueError(f"Text contains unknown characters: {formatted}")
        return [mapping[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""

        decoded: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary.")
            decoded.append(self.vocabulary[token_id])
        return "".join(decoded)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "character", "vocabulary": list(self.vocabulary)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharacterTokenizer:
        if payload.get("type") != "character":
            raise ValueError("Expected a character-tokenizer payload.")
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list) or not all(
            isinstance(token, str) for token in vocabulary
        ):
            raise ValueError("Tokenizer vocabulary must be a list of strings.")
        return cls(vocabulary=tuple(vocabulary))

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tokenizer file must contain a JSON object.")
        return cls.from_dict(payload)

