"""Train my tiny GPT-style model for a few quick classroom-friendly steps."""

import torch

from gpt2_demo.model import CharTokenizer, MiniGPT


SAMPLE_TEXT = (
    "artificial intelligence studies data and markets. "
    "a mock trader can buy sell or hold. "
    "gpt learns patterns in text. "
) * 12

MODEL_SETTINGS = {
    "block_size": 32,
    "embedding_size": 32,
    "num_heads": 4,
    "num_layers": 2,
}


def create_model_and_tokenizer():
    """Create the tokenizer and model from the same tiny text sample."""
    tokenizer = CharTokenizer(SAMPLE_TEXT)
    model = MiniGPT(vocab_size=tokenizer.vocab_size, **MODEL_SETTINGS)
    return model, tokenizer


def get_batch(data, block_size, batch_size):
    """Pick short text slices so the model can practice next-character prediction."""
    start_positions = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[start : start + block_size] for start in start_positions])
    y = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in start_positions]
    )
    return x, y


def train_model(steps=80, show_loss=True):
    """Train the mini GPT model long enough to show the loss going down."""
    torch.manual_seed(42)

    model, tokenizer = create_model_and_tokenizer()
    data = torch.tensor(tokenizer.encode(SAMPLE_TEXT), dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    final_loss = None
    for step in range(steps):
        x_batch, y_batch = get_batch(
            data,
            MODEL_SETTINGS["block_size"],
            batch_size=16,
        )

        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if show_loss and (step == 0 or (step + 1) % 20 == 0):
            print(f"step {step + 1:03d} loss {final_loss:.4f}")

    return model, tokenizer, final_loss


def main():
    """Run the training demo from the terminal."""
    print("=== Mini GPT-2 Training Demo ===")
    print("Training a tiny character-level GPT model on sample text.")
    _, _, final_loss = train_model()
    print(f"Final training loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
