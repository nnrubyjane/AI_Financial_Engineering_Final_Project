"""A very small GPT-style model for educational use.

This is not a full GPT-2 model. It is a tiny character-level transformer that
shows the same main ideas: embeddings, masked self-attention, transformer
blocks, and text generation.
"""

import torch
from torch import nn
import torch.nn.functional as F


class CharTokenizer:
    """Simple character tokenizer for a tiny text dataset."""

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {char: index for index, char in enumerate(self.chars)}
        self.id_to_char = {index: char for index, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text into a list of token IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return "".join(self.id_to_char[token_id] for token_id in token_ids)


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention.

    Q means query: what this position is looking for.
    K means key: what each position contains.
    V means value: the information copied after attention chooses where to look.
    The mask prevents the model from seeing future characters.
    """

    def __init__(self, embedding_size, num_heads, block_size):
        super().__init__()
        if embedding_size % num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.projection = nn.Linear(embedding_size, embedding_size)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch_size, time_steps, embedding_size = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, time_steps, self.num_heads, self.head_size)
        k = k.view(batch_size, time_steps, self.num_heads, self.head_size)
        v = v.view(batch_size, time_steps, self.num_heads, self.head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores / (self.head_size ** 0.5)
        attention_scores = attention_scores.masked_fill(
            self.mask[:, :, :time_steps, :time_steps] == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, time_steps, embedding_size)
        return self.projection(attention_output)


class FeedForward(nn.Module):
    """Small neural network used after attention in each transformer block."""

    def __init__(self, embedding_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """One GPT-style block: attention, feed-forward network, and residual paths."""

    def __init__(self, embedding_size, num_heads, block_size):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embedding_size, num_heads, block_size)
        self.feed_forward = FeedForward(embedding_size)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MiniGPT(nn.Module):
    """A tiny GPT-style language model."""

    def __init__(
        self,
        vocab_size,
        block_size,
        embedding_size=32,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)

        blocks = []
        for _ in range(num_layers):
            blocks.append(TransformerBlock(embedding_size, num_heads, block_size))
        self.blocks = nn.Sequential(*blocks)

        self.final_layer_norm = nn.LayerNorm(embedding_size)
        self.output_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, time_steps = token_ids.shape
        if time_steps > self.block_size:
            raise ValueError("Input is longer than the model block_size.")

        positions = torch.arange(time_steps, device=token_ids.device)
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * time_steps, -1)
            targets_flat = targets.view(batch_size * time_steps)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, token_ids, max_new_tokens):
        """Generate text one token at a time."""
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.block_size :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids

