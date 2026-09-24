# GPT From Scratch

A character-level GPT-style language model implemented from scratch in
PyTorch and trained on the **Tiny Shakespeare** dataset.

This notebook builds the model step by step, starting with character
tokenization and a simple Bigram Language Model, then moving toward the
core components of a decoder-only Transformer: causal self-attention,
multi-head attention, feed-forward layers, layer normalization, residual
connections, positional embeddings, training, validation, and text
generation.

## Project Overview

The goal of this project is to understand **how a GPT-style language
model works internally** instead of using a pre-trained Transformer
library.

The notebook covers:

1.  Dataset loading and exploration
2.  Character-level tokenization
3.  Encoding and decoding
4.  Train/validation split
5.  Batch generation
6.  Next-token prediction
7.  Bigram Language Model baseline
8.  Causal / masked self-attention
9.  Query, Key, and Value projections
10. Scaled dot-product attention
11. Multi-head self-attention
12. Feed-forward network
13. Layer normalization
14. Residual connections
15. Token embeddings
16. Positional embeddings
17. Transformer blocks
18. Cross-entropy loss
19. AdamW optimization
20. Training and validation loss evaluation
21. Autoregressive text generation

## Architecture

``` mermaid
flowchart TD
    A[Raw Shakespeare Text] --> B[Character Vocabulary]
    B --> C[Character Encoder]
    C --> D[Integer Token IDs]
    D --> E[Train / Validation Split]
    E --> F[Input and Target Batches]

    F --> G[Token Embeddings]
    G --> H[Positional Embeddings]
    H --> I[Token + Position]

    I --> J[Transformer Block 1]
    J --> K[Transformer Block 2]
    K --> L[Transformer Block 3]
    L --> M[Transformer Block 4]

    J -.-> J1[LayerNorm]
    J1 --> J2[Causal Multi-Head Self-Attention]
    J2 --> J3[Residual Connection]
    J3 --> J4[LayerNorm]
    J4 --> J5[Feed-Forward Network]
    J5 --> J6[Residual Connection]

    M --> N[Final LayerNorm]
    N --> O[Linear LM Head]
    O --> P[Logits]
    P --> Q[Softmax]
    Q --> R[Next Character]
    R --> S[Autoregressive Generation]
```

## Dataset

The notebook uses the **Tiny Shakespeare** text dataset.

Source used in the notebook:

``` text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The dataset contains approximately **1.1 million characters**.

The notebook reports:

``` text
Dataset length: 1,115,394 characters
Vocabulary size: 65 characters
```

The vocabulary is created from the unique characters appearing in the
dataset.

## Character-Level Tokenization

This project uses **character-level tokenization**.

For example:

``` text
hello
```

is converted into a sequence of integer IDs using two dictionaries:

``` python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

### Encoding

``` python
encode("hello")
```

converts characters into integer token IDs.

### Decoding

``` python
decode(token_ids)
```

converts integer IDs back into characters.

This makes the model's task:

> Given previous characters, predict the next character.

## Train / Validation Split

The encoded dataset is divided into:

-   **90% training data**
-   **10% validation data**

``` python
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]
```

The training set is used to update model parameters, while the
validation set is used to monitor how well the model performs on unseen
text.

## Context Window

The notebook uses:

``` python
block_size = 32
```

in the final model.

This means the model can use up to **32 previous characters** as its
context when making predictions.

During training, an input sequence and a shifted target sequence are
created:

``` text
Input:  A B C D E
Target: B C D E F
```

The model therefore learns next-token prediction.

## Batch Generation

The `get_batch()` function randomly selects sections of the dataset and
creates batches.

``` python
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
```

For the final configuration:

``` text
Batch size  = 16
Context     = 32
```

The resulting tensors have the form:

``` text
x: (batch_size, block_size)
y: (batch_size, block_size)
```

## Bigram Language Model

Before building the Transformer, the notebook creates a simple **Bigram
Language Model**.

A bigram model predicts the next character mainly from the current
character.

The baseline uses an embedding table:

``` python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

It provides a useful starting point for understanding:

-   logits
-   probabilities
-   cross-entropy loss
-   optimization
-   autoregressive generation

The notebook first trains this simpler model and generates text before
moving to self-attention.

## Self-Attention

The notebook then develops self-attention step by step.

For each token representation, three projections are created:

-   **Query (Q)** --- what information the current token is looking for
-   **Key (K)** --- what information each token contains for matching
-   **Value (V)** --- the information that is aggregated

The attention score is calculated using:

``` text
Q × Kᵀ
```

and scaled by the square root of the head dimension.

Conceptually:

``` text
Attention(Q,K,V)
    = softmax(QKᵀ / √dₖ)V
```

## Causal Masking

Because this is a language model, a token must not look at future
tokens.

The notebook creates a lower-triangular mask:

``` python
tril = torch.tril(torch.ones(T, T))
```

Future positions are replaced with negative infinity before softmax:

``` python
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

This produces **causal self-attention**, where each position can attend
only to itself and earlier positions.

## Multi-Head Self-Attention

Instead of using a single attention operation, the model uses multiple
attention heads.

Final configuration:

``` text
Embedding dimension = 64
Number of heads     = 4
Head size           = 16
```

Each head learns different attention patterns.

The outputs of all heads are concatenated and passed through a
projection layer.

``` python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```

## Feed-Forward Network

After attention, each Transformer block contains a feed-forward network.

The notebook uses:

``` python
nn.Linear(n_embd, 4 * n_embd)
nn.ReLU()
nn.Linear(4 * n_embd, n_embd)
```

For an embedding size of 64, the intermediate representation has:

``` text
64 → 256 → 64
```

This provides additional computation for each token after information
has been exchanged through attention.

## Layer Normalization

The notebook explores Layer Normalization and then uses PyTorch's:

``` python
nn.LayerNorm(n_embd)
```

inside the Transformer block.

Layer normalization helps stabilize the representations flowing through
the network.

## Residual Connections

Each Transformer block uses residual connections:

``` python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The structure is:

``` text
Input
  │
  ├── LayerNorm
  │
  ├── Multi-Head Self-Attention
  │
  └── + Residual
        │
        ├── LayerNorm
        │
        ├── Feed-Forward Network
        │
        └── + Residual
```

## Token and Positional Embeddings

The model uses two types of embeddings.

### Token Embedding

``` python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
```

It converts token IDs into vector representations.

### Positional Embedding

``` python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

It gives the model information about the position of each character.

The two representations are added:

``` python
x = tok_emb + pos_emb
```

## Transformer Block

The notebook builds a stack of Transformer blocks.

Final configuration:

  Hyperparameter            Value
  ----------------------- -------
  Batch size                   16
  Context length               32
  Embedding size               64
  Attention heads               4
  Head size                    16
  Transformer layers            4
  Learning rate             0.001
  Training iterations        5000
  Evaluation interval         100
  Evaluation iterations       200
  Dropout                     0.0
  Optimizer                 AdamW

## Model Flow

The final model follows this flow:

``` text
Character IDs
     ↓
Token Embeddings
     +
Positional Embeddings
     ↓
Transformer Block
     ↓
Transformer Block
     ↓
Transformer Block
     ↓
Transformer Block
     ↓
Final LayerNorm
     ↓
Linear Language Model Head
     ↓
Logits over 65 characters
     ↓
Softmax
     ↓
Next-character probability
```

## Loss Function

The model uses **cross-entropy loss**:

``` python
F.cross_entropy(logits, targets)
```

The logits are reshaped before calculating the loss:

``` python
logits = logits.view(B*T, C)
targets = targets.view(B*T)
```

where:

-   `B` = batch size
-   `T` = sequence length
-   `C` = vocabulary size

The objective is to reduce the difference between the predicted next
character and the actual target character.

## Optimizer

The notebook uses:

``` python
torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
```

with:

``` text
learning rate = 1e-3
```

The training loop performs:

``` text
Get batch
   ↓
Forward pass
   ↓
Calculate loss
   ↓
Zero gradients
   ↓
Backpropagation
   ↓
Optimizer step
```

## Training

The final model is trained for:

``` text
5000 iterations
```

The notebook evaluates training and validation loss every:

``` text
100 iterations
```

The reported final evaluation is approximately:

``` text
Step 4999
Train loss: 1.6627
Validation loss: 1.8207
```

The model contains approximately:

``` text
0.209729 million parameters
≈ 210K parameters
```

## Text Generation

After training, the model starts from an initial token and generates
characters one at a time.

The generation process is:

``` text
Current context
      ↓
Transformer
      ↓
Logits for next character
      ↓
Softmax probabilities
      ↓
Sample next character
      ↓
Append character to context
      ↓
Repeat
```

The notebook generates up to:

``` python
max_new_tokens = 2000
```

characters.

The generated output resembles Shakespeare-style dialogue and
formatting, although it is not grammatically or semantically perfect.

## Important Implementation Detail

The final Transformer implementation is stored in a class named:

``` python
BigramLanguageModel
```

The name comes from the earlier baseline implementation, but the final
version contains:

-   token embeddings
-   positional embeddings
-   multiple Transformer blocks
-   causal self-attention
-   multi-head attention
-   feed-forward networks
-   layer normalization
-   residual connections
-   language-model output head

So the final implementation is a **small GPT-style Transformer**,
despite retaining the class name `BigramLanguageModel`.

## Technologies

-   Python
-   PyTorch
-   Google Colab
-   Jupyter Notebook
-   Tiny Shakespeare dataset

## Requirements

The notebook requires Python and PyTorch.

For a local environment:

``` bash
pip install torch
```

For Google Colab, PyTorch is generally already available.

## Running the Notebook

### Google Colab

1.  Open the notebook in Google Colab.
2.  Enable a GPU if available: `Runtime → Change runtime type → GPU`
3.  Run the cells from top to bottom.
4.  The dataset is downloaded automatically.
5.  The model is trained using the configured hyperparameters.
6.  Generated Shakespeare-style text is printed at the end.

### Local Jupyter

Clone/download the project and open:

``` text
GPT_FROM_SCRATCH.ipynb
```

Then run the cells sequentially.

Make sure `input.txt` is available because the final training section
reads:

``` python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

## Project Structure

``` text
GPT_FROM_SCRATCH/
│
├── GPT_FROM_SCRATCH.ipynb
├── input.txt
└── README.md
```

`input.txt` is the Tiny Shakespeare dataset used by the notebook.

## Learning Outcomes

By completing this project, you can understand the main building blocks
behind a GPT-style language model:

-   How text becomes token IDs
-   How training examples are created
-   How next-token prediction works
-   How embeddings represent tokens
-   Why positional information is required
-   How Query, Key, and Value work
-   How causal masking prevents future-token access
-   How self-attention combines information
-   Why multiple attention heads are used
-   How feed-forward networks process representations
-   How LayerNorm and residual connections are used
-   How Transformer blocks are stacked
-   How cross-entropy loss trains a language model
-   How AdamW updates model parameters
-   How autoregressive generation works

## Limitations

This is an educational, small-scale GPT implementation.

It is not intended to compete with large pretrained language models
because:

-   The dataset is small.
-   The vocabulary is character-level.
-   The model has only about 210K parameters.
-   The context length is 32 characters.
-   The model is trained for only 5000 iterations.
-   The generated text is limited in quality and coherence.

The main purpose is to understand the internal mechanics of GPT rather
than achieve production-level language generation.

## Future Improvements

Possible extensions include:

-   Increase the context length.
-   Increase the embedding dimension.
-   Add more Transformer layers.
-   Add more attention heads.
-   Use a larger dataset.
-   Train for more iterations.
-   Add dropout.
-   Add model checkpointing.
-   Save and reload trained weights.
-   Add temperature-based generation.
-   Add top-k sampling.
-   Replace character-level tokenization with a subword tokenizer.
-   Build a simple inference interface.
-   Compare the model with a pretrained GPT model.

## References

The implementation follows the educational approach of building a
GPT-style model step by step and uses the Tiny Shakespeare dataset
associated with Andrej Karpathy's character-level language-model
examples.

Dataset:

``` text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

------------------------------------------------------------------------

## Author

**Esha Alvi**

AI/ML • Generative AI • NLP • LLMs

GitHub: `esha-alvi-ai`
