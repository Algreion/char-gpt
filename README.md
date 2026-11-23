# char-gpt

Character-level, decoder-only transformer implemented in PyTorch without nn.Transformer.

The goal was to create a small GPT-style model by hand to understand and replicate the basic architecture of language models. It was then trained on Shakespeare as a proof-of-concept.

## Description
The model is implemented through multiple classes:
- Data: A shared infrastructure that stores the model's training data and its vocab, and implements some helper functions like splitting datasets for training and validation.
- Modules (Head, MultiHeadAttention, FeedForward, Block): Implementations for the various layers of the model.
- GPT: Decoder-only transformer with a vocab of individual characters. It implements fitting, sampling, custom saving and loading with validation, and a scatter plot of token representations.
- Demo (run_training, run_demo): A small model demo using a Shakespeare dataset.

## Requirements
`pip install torch matplotlib`
- PyTorch
- matplotlib

## Demo
A demo model (1.07M parameters) was trained on a Shakespeare dataset. After 20,000 training steps (11 minutes on a 5070 GPU), it could generate plausible-sounding Shakespeare text.

Sample (cut off after 300 characters):
```
MERCUTIO:
A panting army brother doth not show;
She company my brother loves again.

First Citizen:
Well, he hath got in presently made you beheld his
would I shall be struck in my grave, if she be fear.

MARIANA:
If they have preserved more in your dishonours of courtier,
Which thou weep'st grieved...
```

Run:
- `python transformer.py sample` Samples the model for 300 characters, with 0.8 temperature.
- `python transformer.py sample "ROMEO:"` Samples text beginning with the argument. If a number is passed, it will sample that amount of characters instead.
- `python transformer.py demo` Reproduces the setup for the sample above with seed 42 (PyTorch version 2.9.1+cu128).
- `python transformer.py train [steps]` Retrains a model on the dataset with a default 20,000 steps. The results are saved in models/transformer-shspr-retrained.txt and loss-retrained.png.

Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), which was lightly cleaned up by removing double dashes, the 3 in '3 KING HENRY VI', &c, and fixing the typo 'sha$l'. This reduced the vocabulary from 65 to 62. The cleaned-up dataset is in data/shakespeare.txt.

## Demo - Model
**Architecture**:
- Layers: 6 transformer blocks
- Embedding dimensions: 120
- Attention heads: 6 x 20 dims
- Feed-forward: 120 > 480 > 120, GELU
- Regularization: Dropout 0.05
- Vocabulary: 62 characters (from dataset)
- Positional information: Learned embeddings
- Context: 64
- Optimizer: AdamW, LR 3e-4, batch size 64
- Total parameters: 1,066,862 (`GPT.size()`)

**Training**:
- The model's final losses were `train: 1.19` and `dev: 1.52`.
- Random guessing would have given a loss of ln(62)=4.127. A newly initialized model gets about 4.2.
- Dev loss plateaus after around 10k training steps, while train loss keeps falling.
- The training is approximately reproducible using seed 42.

<img src="loss.png" alt="Loss plot" width="300" />

## Usage as a library
```python
from transformer import GPT, DEVICE

# Training a model on a dataset
g = GPT('./dataset.txt').to(DEVICE)
g.fit(n=10000, plot=True)
g.estimate_loss(batches=500)
g.save('model.txt', 'vocab.txt')

# Load an existing model
g = GPT('./models/vocab.txt', 'models/transformer-shspr.txt').to(DEVICE)

# Sampling
g.sample(100)
g.sample("ROMEO:", temperature=0.8, top_k=40)
```

## Debugging
An earlier version of the model produced much worse text when sampled than what the loss suggested.

This was due to attention being only causal when a training flag was set, meaning that during generation the model's earlier positions attended to later ones, which skewed predictions as it was unlike what the model had seen during training. As the loss used the training flag, it couldn't reveal the issue.

This was fixed by using PyTorch's standard train() and eval(), and always having is_causal=True.
<details>
<summary>Before/after samples</summary>
(seed 100, temperature 1)

Before:

```
Messenger stooping; such my consistraist one, the but is the grief;
With the repul, unnature of the...
```

After:

```
Messenger:
The won; sir, my lord.

POLIXENES:
God be silk,
Profit your brother followers love his ch...
```
</details>

## Known limitations
- The model learns spelling, formatting and local word structures but not meaning. This is expected due to its parameter count and 64-character context, which isn't enough to learn sentence-level structure.
- The vocabulary is fixed to the characters encountered in the training dataset.
- The dev data is the last 10% of the dataset, instead of being a random split.
- Basic training: Training uses a constant learning rate, no gradient clipping or early stopping.
- As the model uses 36 separate head modules and no KV cache, it runs slower than it would with concatenated heads and caching.
- Saving and loading model weights uses a custom format which is less efficient than the PyTorch torch.save.

## Next steps
- A longer context and a larger model.
- A subword tokenizer, and an optimized model using already existing libraries.
- A model trained on a larger dataset, such as TinyStories, capable of meaningful output.

## Acknowledgements
The architecture follows the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper, and the structure was inspired by the [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture. The implementation is my own.