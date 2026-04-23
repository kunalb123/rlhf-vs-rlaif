"""
Part 2a: Train a reward model on human preference pairs from HH-RLHF.

Loads the merged SFT checkpoint, adds a scalar classification head, and fine-tunes
with TRL's RewardTrainer on (chosen, rejected) response pairs.

Run from project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlhf.train_reward_model
"""

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from src.utils import extract_prompt_and_response, get_device, load_config, load_hh_rlhf


def prepare_reward_dataset(raw_dataset) -> Dataset:
    examples = []
    for item in raw_dataset:
        chosen_prompt, chosen_resp = extract_prompt_and_response(item["chosen"])
        _, rejected_resp = extract_prompt_and_response(item["rejected"])

        if not chosen_resp or not rejected_resp:
            continue

        examples.append(
            {
                "chosen": chosen_prompt + chosen_resp,
                "rejected": chosen_prompt + rejected_resp,
            }
        )
    return Dataset.from_list(examples)


def train_reward_model(config_path: str = "configs/training_config.yaml") -> None:
    torch.mps.empty_cache()
    config = load_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    sft_ckpt = "checkpoints/sft/final"
    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # SFT base + scalar head (num_labels=1 → single reward score)
    model = AutoModelForSequenceClassification.from_pretrained(
        sft_ckpt,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    raw_dataset = load_hh_rlhf("train", num_samples=config["data"]["num_samples"])
    train_dataset = prepare_reward_dataset(raw_dataset)
    print(f"Training on {len(train_dataset)} preference pairs")

    rm_cfg = config["reward_model"]
    reward_config = RewardConfig(
        output_dir="checkpoints/reward_model",
        num_train_epochs=rm_cfg["num_epochs"],
        per_device_train_batch_size=config["reward_model"].get("batch_size", config["training"]["batch_size"]),
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=rm_cfg["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        max_length=rm_cfg["max_length"],
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=500,
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model("checkpoints/reward_model/final")
    tokenizer.save_pretrained("checkpoints/reward_model/final")
    print("Reward model done — saved to checkpoints/reward_model/final")


if __name__ == "__main__":
    import sys
    train_reward_model(sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml")
