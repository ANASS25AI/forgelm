# ForgeLM milestone roadmap

Each milestone must add theory, tested code, an experiment and a short report.
Large model weights and datasets are never committed to Git.

| ID | Milestone | Primary deliverable |
|---:|---|---|
| 01 | Character language model | Bigram baseline and complete training loop |
| 02 | Tokenization | Byte and BPE tokenizers from scratch |
| 03 | Gradient foundations | Small autograd engine and gradient checking |
| 04 | Neural language model | Embeddings, MLP and logits |
| 05 | Attention | Causal single-head and multi-head attention |
| 06 | Transformer | Complete decoder-only model |
| 07 | Pretraining engine | Optimizer, schedules, precision and checkpoints |
| 08 | Small-model study | Train and evaluate 5M–100M parameter models |
| 09 | GPU optimization | Profiling, compilation and fused operations |
| 10 | Distributed training | DDP, FSDP, TP, PP and distributed checkpoints |
| 11 | Basic inference | Sampling, prefill, decode and KV cache |
| 12 | Serving runtime | Continuous batching and paged KV management |
| 13 | Advanced inference | Quantization, speculative decoding and kernels |
| 14 | Deployment | Streaming API, containers, metrics and load tests |
| 15 | Instruction tuning | Chat formatting and supervised fine-tuning |
| 16 | Preference learning | Reward modeling and DPO |
| 17 | Reasoning RL | Verifiable rewards and policy optimization |
| 18 | Test-time reasoning | Search, self-consistency and verifiers |
| 19 | Tool calling | Structured actions and recovery |
| 20 | Agent runtime | Planning, memory, safety and evaluation |
| 21 | Vision encoder | Patch embeddings and Vision Transformer |
| 22 | Multimodal model | Projectors, fusion and image-text training |
| 23 | Paper reproduction | Reproduce a relevant published result |
| 24 | Original research | Hypothesis, ablations and technical report |

## Milestone acceptance gate

A milestone is complete only when:

1. the mathematics and tensor shapes are documented;
2. unit tests pass;
3. the end-to-end smoke test passes;
4. linting and type checking pass;
5. an experiment is reproducible from a configuration;
6. results and limitations are recorded;
7. CI passes on the pull request;
8. the milestone is merged and tagged.

