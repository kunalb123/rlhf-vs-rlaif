#!/usr/bin/env bash
# Quick end-to-end smoke test using tiny data.
# Overrides num_samples and num_eval_samples via a temp config.
# Expected runtime: ~5-10 minutes total.
set -euo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Write a minimal config for testing
SMOKE_CONFIG="configs/smoke_test_config.yaml"
cat > "$SMOKE_CONFIG" << 'EOF'
model:
  name: "gpt2-medium"
  max_length: 64

training:
  batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  num_epochs: 1
  warmup_steps: 0

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["c_attn", "c_proj"]

data:
  dataset: "Anthropic/hh-rlhf"
  num_samples: 8
  grpo_num_samples: 8
  max_prompt_length: 32
  max_response_length: 32

reward_model:
  batch_size: 1
  num_epochs: 1
  learning_rate: 2.0e-5
  max_length: 64

grpo:
  learning_rate: 1.0e-6
  batch_size: 2
  gradient_accumulation_steps: 1
  num_generations: 2
  num_epochs: 1
  max_new_tokens: 32
  temperature: 1.0
  kl_coef: 0.2
  max_grad_norm: 1.0

rlaif:
  judge_model: "claude-haiku-4-5-20251001"

evaluation:
  num_eval_samples: 4
  output_dir: "results/smoke_test"
EOF

echo "=== Smoke test config written to $SMOKE_CONFIG ==="
echo ""

echo "=== [1/5] SFT ==="
python -m src.sft.train_sft "$SMOKE_CONFIG"

echo "=== [2/5] Reward Model ==="
python -m src.rlhf.train_reward_model "$SMOKE_CONFIG"

echo "=== [3/5] RLHF PPO ==="
python -m src.rlhf.train_ppo_rlhf "$SMOKE_CONFIG"

echo "=== [4/5] RLAIF PPO ==="
python -m src.rlaif.train_ppo_rlaif "$SMOKE_CONFIG"

echo "=== [5/5] Evaluation ==="
python -m src.evaluation.evaluate "$SMOKE_CONFIG"
python -m src.evaluation.reward_hacking

echo ""
echo "=== Smoke test passed! ==="
echo "Results in results/smoke_test/"
