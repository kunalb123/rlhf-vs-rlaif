"""
Part 2b: GRPO fine-tuning with the learned reward model (RLHF).

Uses TRL's GRPOTrainer (the modern replacement for the classic PPOTrainer).
GRPO generates num_generations responses per prompt, scores them all, then
updates the policy using group-relative rewards. KL divergence is tracked
correctly against a frozen reference copy of the model.

Run from project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlhf.train_ppo_rlhf
"""

import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.utils import extract_prompt_and_response, get_device, load_config, load_hh_rlhf


def prepare_grpo_dataset(raw_dataset) -> Dataset:
    prompts = []
    for item in raw_dataset:
        prompt, _ = extract_prompt_and_response(item["chosen"])
        if prompt:
            prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts)


def _rep_rate(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams)


def build_reward_fn(reward_model, tokenizer, max_length: int):
    """Returns a reward function that scores prompt+completion pairs."""
    # Reward model stays on CPU to avoid OOM alongside the policy on MPS
    reward_model.to("cpu")
    reward_model.eval()

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        full_texts = [p + c for p, c in zip(prompts, completions)]
        inputs = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = reward_model(**inputs).logits.squeeze(-1)
        # Subtract repetition penalty before normalizing — discourages the reward
        # hacking pattern of producing long repetitive "I'm not sure" completions.
        rep_penalties = torch.tensor([_rep_rate(c) for c in completions])
        scores = scores - 2.0 * rep_penalties
        # Center then tanh so GPT-2's negative logits land near 0 where tanh has
        # curvature, preserving within-group differences.
        return torch.tanh(scores - scores.mean()).tolist()

    return reward_fn


def train_ppo_rlhf(config_path: str = "configs/training_config.yaml") -> None:
    torch.mps.empty_cache()
    config = load_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft/final")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained("checkpoints/sft/final")

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "checkpoints/reward_model/final", num_labels=1
    )
    reward_fn = build_reward_fn(reward_model, tokenizer, config["reward_model"]["max_length"])

    grpo_config = GRPOConfig(
        output_dir="checkpoints/rlhf_ppo",
        num_train_epochs=config["grpo"]["num_epochs"],
        per_device_train_batch_size=config["grpo"]["batch_size"],
        gradient_accumulation_steps=config["grpo"]["gradient_accumulation_steps"],
        learning_rate=config["grpo"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        num_generations=config["grpo"]["num_generations"],
        max_completion_length=config["grpo"]["max_new_tokens"],
        temperature=config["grpo"]["temperature"],
        beta=config["grpo"]["kl_coef"],
        max_grad_norm=config["grpo"]["max_grad_norm"],
        logging_steps=10,
        save_steps=500,
        report_to="none",
    )

    raw_dataset = load_hh_rlhf("train", num_samples=config["data"].get("grpo_num_samples", config["data"]["num_samples"]))
    train_dataset = prepare_grpo_dataset(raw_dataset)
    print(f"Training on {len(train_dataset)} prompts, {config['grpo']['num_generations']} generations each")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model("checkpoints/rlhf_ppo/final")
    tokenizer.save_pretrained("checkpoints/rlhf_ppo/final")
    print("RLHF GRPO done — saved to checkpoints/rlhf_ppo/final")


if __name__ == "__main__":
    import sys
    train_ppo_rlhf(sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml")
