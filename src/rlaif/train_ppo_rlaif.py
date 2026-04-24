"""
Part 3: GRPO fine-tuning with Claude as the reward signal (RLAIF).

Identical structure to RLHF GRPO, but the reward function calls the Claude API
instead of a learned reward model. Claude scores each response on helpfulness
and conciseness (1-10), which is normalized to [-1, 1] before being passed to GRPO.

Run from project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlaif.train_ppo_rlaif
"""

import os
import re

import anthropic
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.utils import extract_prompt_and_response, get_device, load_config, load_hh_rlhf

RANK_PROMPT = """\
Rank these {n} AI assistant responses to the same prompt from best to worst. \
Prefer responses that are relevant and concise over verbose or repetitive ones.

Prompt: {prompt}

{responses}

Output ONLY comma-separated numbers, best first. No other text. \
Example for 4 responses: 3,1,4,2"""


def _rank_group(
    client: anthropic.Anthropic, prompt: str, completions: list[str], model: str
) -> list[int] | None:
    """Ask Claude to rank completions. Returns list where ranks[i] = rank of completion i (1=best)."""
    n = len(completions)
    if n == 1:
        return None
    responses_text = "\n\n".join(
        f"Response {i + 1}: {c}" for i, c in enumerate(completions)
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=20,
            messages=[
                {
                    "role": "user",
                    "content": RANK_PROMPT.format(
                        n=n, prompt=prompt, responses=responses_text
                    ),
                }
            ],
        )
        text = msg.content[0].text.strip()
        all_nums = [int(x) for x in re.findall(r'\b\d+\b', text) if 1 <= int(x) <= n]
        seen: set[int] = set()
        ranks = []
        for x in all_nums:
            if x not in seen:
                seen.add(x)
                ranks.append(x)
        if len(ranks) != n or set(ranks) != set(range(1, n + 1)):
            return None
        return ranks
    except Exception as e:
        print(f"[RLAIF rank error] {type(e).__name__}: {e}")
        return None


def prepare_grpo_dataset(raw_dataset) -> Dataset:
    prompts = []
    for item in raw_dataset:
        prompt, _ = extract_prompt_and_response(item["chosen"])
        if prompt:
            prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts)


def build_reward_fn(judge_model: str):
    """Returns a reward function that ranks completions per prompt using Claude."""
    client = anthropic.Anthropic()

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        # GRPO passes the same prompt repeated num_generations times.
        # Group completions by prompt, rank within each group, then convert
        # rank → reward so GRPO always sees non-zero within-group variance.
        groups: dict[str, dict] = {}
        prompt_order: list[str] = []
        for i, (p, c) in enumerate(zip(prompts, completions)):
            if p not in groups:
                groups[p] = {"indices": [], "completions": []}
                prompt_order.append(p)
            groups[p]["indices"].append(i)
            groups[p]["completions"].append(c)

        rewards = [0.0] * len(completions)
        for prompt in prompt_order:
            group = groups[prompt]
            n = len(group["completions"])
            ranks = _rank_group(client, prompt, group["completions"], judge_model)
            if ranks is None:
                continue  # leave neutral 0.0 for this group
            for pos, idx in enumerate(group["indices"]):
                # rank 1 (best) → +1.0, rank n (worst) → -1.0, linearly spaced
                rewards[idx] = 1.0 - 2.0 * (ranks[pos] - 1) / (n - 1)
        return rewards

    return reward_fn


def train_ppo_rlaif(config_path: str = "configs/training_config.yaml") -> None:
    torch.mps.empty_cache()
    config = load_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    judge_model = config["rlaif"]["judge_model"]

    # Verify API connectivity before starting training
    print("Testing Claude API connection...")
    _test_client = anthropic.Anthropic()
    _test_msg = _test_client.messages.create(
        model=judge_model, max_tokens=5,
        messages=[{"role": "user", "content": "Reply with the number 1."}]
    )
    print(f"API test OK: '{_test_msg.content[0].text.strip()}'")

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
