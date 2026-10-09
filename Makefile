.PHONY: install lint typecheck test quality train-stage-01

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

quality: lint typecheck test

train-stage-01:
	python -m forgelm.training.train_bigram --input data/sample_corpus.txt --steps 500

