# Stage 01 experiment report

## Question

Can the smallest complete language-model pipeline learn next-character
statistics from the included miniature corpus?

## Configuration

| Setting | Value |
|---|---:|
| Device | CPU |
| Vocabulary size | 28 |
| Batch size | 32 |
| Sequence length | 8 |
| Optimizer | AdamW |
| Learning rate | 0.01 |
| Training steps | 500 |
| Random seed | 1337 |

## Results

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 0 | 3.8652 | 3.7869 |
| 100 | 2.9400 | 2.9727 |
| 200 | 2.4519 | 2.5530 |
| 300 | 2.2483 | 2.3986 |
| 400 | 2.1706 | 2.3492 |
| 500 | 2.0947 | 2.3305 |

Both losses decreased, demonstrating that the data, forward pass, loss,
backpropagation and optimizer are connected correctly.

## Interpretation

The model learned which characters commonly follow each character. Generated
samples contain fragments that resemble English spelling and punctuation, but
they do not preserve words, grammar or meaning. This is expected because the
bigram model conditions only on the immediately preceding character.

## Limitation

The sample corpus is intentionally tiny and repetitive. These numbers verify
the software pipeline; they are not evidence of useful language understanding.

## Next experiment

Implement byte and BPE tokenization, then compare compression, vocabulary size
and unknown-token behavior before replacing the bigram with a neural model.

