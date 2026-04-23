"""
Merge a LoRA SFT checkpoint into a plain causal LM checkpoint.

Usage:
    python scripts/merge_sft_checkpoint.py <lora_checkpoint_dir> <output_dir>

Example:
    python scripts/merge_sft_checkpoint.py \
        checkpoints/sft/checkpoint-114 \
        checkpoints/sft/checkpoint_114_merged
"""

import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

lora_path, out_path = sys.argv[1], sys.argv[2]

print(f"Loading base model...")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained(lora_path)

print(f"Loading LoRA adapter from {lora_path}...")
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {out_path}...")
model.save_pretrained(out_path)
tokenizer.save_pretrained(out_path)
print("Done.")
