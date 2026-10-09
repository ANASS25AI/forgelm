"""Command-line entry point for ForgeLM Stage 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from forgelm.data import split_tokens
from forgelm.tokenization import CharacterTokenizer
from forgelm.training.bigram_trainer import BigramTrainingConfig, train_bigram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ForgeLM Stage 1 bigram model.")
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 training text file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage_01_bigram"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--generate-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.train(text)
    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_tokens, validation_tokens = split_tokens(all_tokens)

    shortest_split = min(len(train_tokens), len(validation_tokens))
    if shortest_split <= args.block_size:
        raise ValueError(
            "The corpus is too short for this block size after the train/validation split."
        )

    config = BigramTrainingConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    model, history = train_bigram(
        train_tokens,
        validation_tokens,
        vocab_size=tokenizer.vocab_size,
        config=config,
        device=device,
    )

    newline_id = tokenizer.token_to_id.get("\n", 0)
    prompt = torch.tensor([[newline_id]], dtype=torch.long, device=device)
    sample_ids = model.generate(
        prompt,
        max_new_tokens=args.generate_tokens,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )[0].tolist()
    sample = tokenizer.decode(sample_ids)

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output / "tokenizer.json")
    torch.save(model.state_dict(), args.output / "model.pt")
    save_json(args.output / "metrics.json", history)
    save_json(
        args.output / "run_config.json",
        {
            **config.__dict__,
            "device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": len(train_tokens),
            "validation_tokens": len(validation_tokens),
        },
    )
    (args.output / "sample.txt").write_text(sample, encoding="utf-8")

    for record in history:
        print(
            f"step={record['step']:>5} "
            f"train_loss={record['train_loss']:.4f} "
            f"validation_loss={record['validation_loss']:.4f}"
        )
    print(f"device={device} vocab_size={tokenizer.vocab_size}")
    print("\nGenerated sample:\n")
    print(sample)


if __name__ == "__main__":
    main()

