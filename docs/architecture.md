# Architecture principles

## One evolving system

ForgeLM is one codebase. New stages reuse stable interfaces rather than
replacing the entire project. For example, the bigram model and future
Transformer both accept token-ID tensors and produce vocabulary logits.

## From-scratch and reference tracks

Important ideas are first implemented in their simplest correct form. A later
reference implementation uses mature PyTorch systems and is benchmarked against
the educational implementation. This separates conceptual understanding from
production optimization without losing either.

## Reproducibility

Every experiment records its configuration, random seed, code revision,
metrics and generated samples. Large datasets, checkpoints and runtime outputs
are stored outside Git and referenced through lightweight metadata.

## Cost discipline

CPU smoke tests and tiny models verify correctness. A single GPU is used for
meaningful model training. Multiple GPUs are reserved for experiments that
specifically teach or measure distributed behavior. Expensive runs require a
written compute and cost estimate first.

