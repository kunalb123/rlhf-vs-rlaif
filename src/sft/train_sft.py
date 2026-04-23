"""
Part 1: Supervised Fine-Tuning baseline.

Trains GPT-2 medium with LoRA on the "chosen" responses from HH-RLHF, then merges
LoRA weights into the base model before saving so downstream stages load cleanly.

Early stopping is used to automatically find the right number of epochs: eval loss
is checked after each epoch and training stops when it hasn't improved for
`sft.early_stopping_patience` consecutive epochs (default 2).

Run from project root:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.sft.train_sft
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from src.utils import extract_prompt_and_response, get_device, load_config, load_hh_rlhf


def prepare_sft_dataset(raw_dataset, max_length: int) -> Dataset:
    examples = []
    for item in raw_dataset:
        prompt, response = extract_prompt_and_response(item["chosen"])
        if response:
            examples.append({"text": prompt + response})
    return Dataset.from_list(examples)


def train_sft(config_path: str = "configs/training_config.yaml") -> None:
    torch.mps.empty_cache()
    config = load_config(config_path)
    device = get_device()
    print(f"Device: {device}")

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_hh_rlhf("train", num_samples=config["data"]["num_samples"])
    full_dataset = prepare_sft_dataset(raw_dataset, config["model"]["max_length"])

    sft_cfg = config.get("sft", {})
    eval_split = sft_cfg.get("eval_split", 0.1)
    patience = sft_cfg.get("early_stopping_patience", 2)

    split = full_dataset.train_test_split(test_size=eval_split, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)}")

    sft_config = SFTConfig(
        output_dir="checkpoints/sft",
        num_train_epochs=sft_cfg.get("max_epochs", 10),
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        max_length=config["model"]["max_length"],
        dataset_text_field="text",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()

    # Merge LoRA into base weights so downstream stages load as a plain causal LM
    merged = model.merge_and_unload()
    merged.save_pretrained("checkpoints/sft/final")
    tokenizer.save_pretrained("checkpoints/sft/final")
    print("SFT done — saved merged model to checkpoints/sft/final")


if __name__ == "__main__":
    import sys
    train_sft(sys.argv[1] if len(sys.argv) > 1 else "configs/training_config.yaml")
