"""Generate a short text sample with the tiny GPT-style model."""

from pathlib import Path

import torch

from gpt2_demo.train_demo import MODEL_SETTINGS, train_model


def save_sample_output(output_text):
    """Save the generation output to the results folder."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "results" / "gpt_sample_output.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output_text + "\n", encoding="utf-8")


def run_generation_demo():
    """Train briefly, then generate a small text sample."""
    torch.manual_seed(42)
    model, tokenizer, _ = train_model(steps=80, show_loss=False)
    model.eval()

    prompt = "ai"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        generated_ids = model.generate(prompt_ids, max_new_tokens=80)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    output_text = "\n".join(
        [
            "=== Mini GPT-2 Text Generation Demo ===",
            f"Block size: {MODEL_SETTINGS['block_size']}",
            f"Prompt: {prompt}",
            f"Generated text: {generated_text}",
        ]
    )
    save_sample_output(output_text)
    return output_text


def main():
    """Run the generation demo."""
    print(run_generation_demo())


if __name__ == "__main__":
    main()

