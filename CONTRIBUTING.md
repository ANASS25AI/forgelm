# Git and GitHub workflow

Every ForgeLM milestone is also a Git and CI/CD lesson.

## Branch workflow

```bash
git switch main
git pull --ff-only
git switch -c milestone/01-bigram
```

Make small commits while learning:

```bash
git add src/forgelm/tokenization tests/unit/test_character_tokenizer.py
git commit -m "feat(tokenizer): add character tokenizer"

git add src/forgelm/data tests/unit/test_batches.py
git commit -m "feat(data): add next-token batch builder"

git add src/forgelm/models tests/unit/test_bigram.py
git commit -m "feat(model): add bigram language model"
```

Before pushing:

```bash
ruff check .
mypy src
pytest
```

Then push and open a pull request:

```bash
git push -u origin milestone/01-bigram
```

Merge only after CI passes. Tag stable milestones:

```bash
git switch main
git pull --ff-only
git tag -a stage-01 -m "ForgeLM Stage 01: bigram language model"
git push origin stage-01
```

## Commit prefixes

- `feat`: a new capability
- `fix`: a bug correction
- `test`: test-only changes
- `docs`: explanations or documentation
- `perf`: measured performance improvement
- `refactor`: internal change with equivalent behavior
- `ci`: automation and delivery changes
- `chore`: maintenance

## Pull-request rule

A pull request must explain what changed, why it is correct, how it was tested,
its measured result, and any remaining limitation.

