import os
import torch
import yaml
from datasets import load_dataset
from typing import Optional


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_prompt_and_response(text: str) -> tuple[str, str]:
    """Parse the final Human/Assistant turn from HH-RLHF conversation format."""
    # Format: "\n\nHuman: ...\n\nAssistant: ..."
    parts = text.strip().split("\n\nAssistant:")
    if len(parts) < 2:
        return text, ""
    response = parts[-1].strip()
    human_parts = parts[-2].split("\n\nHuman:")
    prompt = "\n\nHuman:" + human_parts[-1] + "\n\nAssistant:"
    return prompt, response


def load_hh_rlhf(split: str = "train", num_samples: Optional[int] = None):
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def compute_response_stats(responses: list[str]) -> dict:
    """Return basic stats used for reward hacking detection."""
    lengths = [len(r.split()) for r in responses]

    def repetition_rate(text: str, n: int = 4) -> float:
        words = text.split()
        if len(words) < n:
            return 0.0
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        return 1 - len(set(ngrams)) / len(ngrams)

    rep_rates = [repetition_rate(r) for r in responses]

    return {
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "mean_repetition_rate": sum(rep_rates) / len(rep_rates),
    }
