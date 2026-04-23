"""
Part 3: GRPO fine-tuning with Claude as the reward signal (RLAIF).

Identical structure to RLHF GRPO, but the reward function calls the Claude API
instead of a learned reward model. Claude scores each response on helpfulness
and conciseness (1-10), which is normalized to [-1, 1] before being passed to GRPO.

Run from project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlaif.train_ppo_rlaif
"""

import os

import anthropic
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.utils import extract_prompt_and_response, get_device, load_config, load_hh_rlhf

JUDGE_PROMPT = """\
Evaluate this AI assistant response on two criteria:
- Helpfulness: does it directly address the question with useful information?
- Conciseness: is it appropriately brief without padding?

Prompt: {prompt}
Response: {response}

Reply with a single integer from 1 (poor) to 10 (excellent). No explanation."""


def get_claude_score(client: anthropic.Anthropic, prompt: str, response: str, model: str) -> float:
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(prompt=prompt, response=response),
                }
            ],
        )
        return max(1.0, min(10.0, float(msg.content[0].text.strip())))
    except Exception:
        return 5.0  # neutral fallback on API error


def prepare_grpo_dataset(raw_dataset) -> Dataset:
    prompts = []
    for item in raw_dataset:
        prompt, _ = extract_prompt_and_response(item["chosen"])
        if prompt:
            prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts)


def build_reward_fn(judge_model: str):
    """Returns a reward function that scores responses using the Claude API."""
    client = anthropic.Anthropic()

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        raw_scores = [
            get_claude_score(client, p, c, judge_model)
            for p, c in zip(prompts, completions)
        ]
        # Normalize Claude's 1-10 scale to roughly [-1, 1] for GRPO
        return [(s - 5.5) / 4.5 for s in raw_scores]

    return reward_fn


def train_ppo_rlaif(config_path: str = "configs/training_config.yaml") -> None:
    torch.mps.empty_cache()
    config = load_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    judge_model = config["rlaif"]["judge_model"]
    reward_fn = build_reward_fn(judge_model)

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/sft/final")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained("checkpoints/sft/final")

    grpo_config = GRPOConfig(
        output_dir="checkpoints/rlaif_ppo",
        num_train_epochs=config["grpo"]["num_epochs"],
        per_device_train_batch_size=config["grpo"]["batch_size"],
        gradient_accumulation_steps=config["grpo"]["gradient_accumulation_steps"],
        learning_rate=config["grpo"]["learning_rate"],
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

    trainer.save_model("checkpoints/rlaif_ppo/final")
    tokenizer.save_pretrained("checkpoints/rlaif_ppo/final")
    print("RLAIF GRPO done — saved to checkpoints/rlaif_ppo/final")


if __name__ == "__main__":
    import sys
    train_ppo_rlaif(sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml")
