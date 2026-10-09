# Stage 1 — Character-level bigram language model

## Learning objectives

By the end of this stage, you should be able to explain how text becomes token
IDs, how input-target pairs are formed, how a model produces logits, how
cross-entropy measures prediction error and how gradient descent changes model
parameters.

## 1. Tokenization

Suppose the text is `hello`. A character tokenizer may assign:

| Character | Token ID |
|---|---:|
| `e` | 0 |
| `h` | 1 |
| `l` | 2 |
| `o` | 3 |

The text becomes:

\[
[1,0,2,2,3].
\]

These are categorical identifiers. Token ID 3 is not numerically “larger” or
more important than token ID 1.

## 2. Inputs and next-token targets

Given token IDs

\[
[x_1,x_2,x_3,x_4,x_5],
\]

a four-token training example uses

\[
X=[x_1,x_2,x_3,x_4]
\]

as the input and

\[
Y=[x_2,x_3,x_4,x_5]
\]

as the target. The target is the same sequence shifted left by one position.

In code, a batch of inputs has shape

\[
[B,T],
\]

where:

- \(B\), read “capital B,” is the batch size;
- \(T\), read “capital T,” is the sequence length;
- every element is an integer token ID.

## 3. Bigram model

A bigram model uses only the current token to predict the next token:

\[
P(x_{t+1}\mid x_t).
\]

Read this as: “the probability of the token at position \(t+1\), given the
token at position \(t\).”

The model owns a trainable matrix

\[
W\in\mathbb{R}^{V\times V}.
\]

Notation:

- \(W\) is the parameter matrix;
- \(\in\) means “belongs to”;
- \(\mathbb{R}\) is the set of real numbers;
- \(V\) is the vocabulary size;
- \(V\times V\) means \(V\) rows and \(V\) columns.

Row \(i\) contains the output scores used when the current token ID is \(i\).
PyTorch's `nn.Embedding(V, V)` performs exactly this row lookup.

For an input tensor of shape \([B,T]\), the output logits have shape:

\[
[B,T,V].
\]

A logit is an unrestricted real-valued score. It is not yet a probability.

## 4. Softmax

For logits \(z_1,\ldots,z_V\), softmax produces:

\[
p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}.
\]

Read it as: “the probability of token \(i\) equals the exponential of its
logit divided by the sum of the exponentials of all vocabulary logits.”

- \(p_i\): predicted probability of vocabulary item \(i\);
- \(z_i\): its logit;
- \(e\): Euler's number, approximately 2.718;
- \(\sum\): add the following terms;
- \(j\): an index that visits every vocabulary item.

Softmax makes every probability positive and makes all probabilities sum to 1.

## 5. Cross-entropy loss

For one correct target token \(y\), the loss is:

\[
\ell=-\log p_y.
\]

Read it as: “the loss equals the negative natural logarithm of the probability
assigned to the correct token.”

- \(\ell\), read “ell,” is the loss for one prediction;
- \(p_y\) is the probability assigned to the correct target;
- \(\log\) is the natural logarithm.

If the correct token receives high probability, the loss is small. If it
receives low probability, the loss is large.

For all positions in a batch:

\[
\mathcal{L}
=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log p_{b,t,y_{b,t}}.
\]

Here \(\mathcal{L}\) is one scalar: the mean loss over all \(B\times T\)
predictions.

## 6. Training step

One optimizer step performs:

1. sample input and target batches;
2. calculate logits;
3. calculate cross-entropy;
4. calculate gradients with `loss.backward()`;
5. update parameters with `optimizer.step()`.

The bigram model cannot understand long context, meaning or grammar. Its job is
to establish a correct baseline and stable interfaces for the Transformer that
will replace it later.

## Exercises

1. For the sequence `[4, 7, 2, 9, 3]`, write the four-token input and target.
2. If \(B=32\), \(T=8\), and \(V=28\), give the input, target and logit shapes.
3. Calculate \(-\log(0.8)\) and \(-\log(0.1)\). Which prediction is better?
4. Explain why a bigram model cannot distinguish `the bank of a river` from
   `the bank approved a loan` using distant context.
5. Change the temperature during generation and describe the difference.

