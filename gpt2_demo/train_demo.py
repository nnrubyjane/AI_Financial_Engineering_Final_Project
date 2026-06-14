"""Train my tiny GPT-style model on a custom finance-and-AI dataset."""

from pathlib import Path

import torch

from gpt2_demo.model import CharTokenizer, MiniGPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "finance_ai_dataset.txt"

MODEL_SETTINGS = {
    "block_size": 32,
    "embedding_size": 32,
    "num_heads": 4,
    "num_layers": 2,
}


def load_training_text():
    """Load the small custom dataset used for this project."""
    return DATASET_PATH.read_text(encoding="utf-8")


def create_model_and_tokenizer():
    """Create the tokenizer and model from the custom dataset."""
    training_text = load_training_text()
    tokenizer = CharTokenizer(training_text)
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


def train_model(steps=60, show_loss=True):
    """Train the mini GPT model long enough to show the loss going down."""
    torch.manual_seed(42)

    training_text = load_training_text()
    model, tokenizer = create_model_and_tokenizer()
    data = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
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
    print(f"Dataset: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print("Dataset note: this is a custom finance-and-AI dataset, not Tiny Shakespeare.")
    print("Training a tiny character-level GPT model on the custom dataset.")
    _, _, final_loss = train_model()
    print(f"Final training loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
