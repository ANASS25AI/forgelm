# ForgeLM

ForgeLM is a long-term, educational and research-oriented project for building
the complete language-model stack from first principles.

The same repository will grow through these systems:

1. tokenization and language-model data;
2. a decoder-only Transformer;
3. pretraining and evaluation;
4. distributed training;
5. an optimized inference engine;
6. production serving and deployment;
7. reasoning-oriented post-training;
8. an evaluated agent runtime;
9. a multimodal language-and-vision model;
10. reproducible research experiments.

## Current status

**Stage 1 — character tokenizer and bigram language model**

The current milestone implements the smallest complete language-model pipeline:

```text
text -> token IDs -> training batches -> logits -> cross-entropy
     -> gradients -> optimizer update -> generated text
```

Read [`docs/lessons/stage_01_bigram.md`](docs/lessons/stage_01_bigram.md)
before reading the implementation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest
python -m forgelm.training.train_bigram \
  --input data/sample_corpus.txt \
  --steps 500
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Full repository architecture

| Path | Long-term responsibility |
|---|---|
| `.github/` | CI, pull requests, issue templates and later releases |
| `configs/` | Versioned experiment and deployment configurations |
| `data/` | Small tracked examples; large raw/processed data remains untracked |
| `docs/` | Architecture, lessons, mathematical notes and decisions |
| `experiments/` | Reproducible experiment definitions and reports |
| `scripts/` | Thin operational entry points |
| `src/forgelm/tokenization/` | Character, byte and BPE tokenizers |
| `src/forgelm/data/` | Cleaning, packing, sharding and distributed loading |
| `src/forgelm/models/` | Bigram, Transformer and multimodal model components |
| `src/forgelm/training/` | Optimization, schedules, checkpoints and pretraining |
| `src/forgelm/distributed/` | DDP, FSDP, tensor, pipeline and context parallelism |
| `src/forgelm/inference/` | Decoding, KV cache, batching and scheduling |
| `src/forgelm/kernels/` | Triton and other optimized kernels |
| `src/forgelm/serving/` | API server, streaming, metrics and deployment runtime |
| `src/forgelm/evaluation/` | Quality, reasoning and systems evaluation |
| `src/forgelm/posttraining/` | SFT, preference optimization and RL |
| `src/forgelm/reasoning/` | Verifiers, search and test-time computation |
| `src/forgelm/agents/` | Tool use, memory, planning and agent evaluation |
| `src/forgelm/multimodal/` | Vision encoders, projectors and multimodal fusion |
| `src/forgelm/utils/` | Shared reproducibility and observability utilities |
| `tests/unit/` | Fast mathematical and component-level tests |
| `tests/integration/` | End-to-end and later distributed smoke tests |

See [`docs/roadmap.md`](docs/roadmap.md) for the milestone sequence and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the Git/GitHub learning workflow.

