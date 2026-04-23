#!/usr/bin/env bash
# Run all three training stages + evaluation in sequence.
# Prerequisites: pip install -r requirements.txt && cp .env.example .env (add API key)
#
# Usage:
#   bash scripts/run_all.sh [--skip-sft] [--skip-reward-model] [--skip-rlhf] [--skip-rlaif] [--skip-eval]
set -euo pipefail

SKIP_SFT=false
SKIP_REWARD=false
SKIP_RLHF=false
SKIP_RLAIF=false
SKIP_EVAL=false

for arg in "$@"; do
  case $arg in
    --skip-sft)          SKIP_SFT=true ;;
    --skip-reward-model) SKIP_REWARD=true ;;
    --skip-rlhf)         SKIP_RLHF=true ;;
    --skip-rlaif)        SKIP_RLAIF=true ;;
    --skip-eval)         SKIP_EVAL=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

purge_mps() {
  python -c "import torch; torch.mps.empty_cache()" 2>/dev/null || true
}

skip_note() {
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Skipped: $1 ==="
}

if [ "$SKIP_SFT" = true ]; then
  skip_note "Part 1: SFT (using checkpoint: checkpoints/sft/final)"
else
  echo "=== Part 1: SFT Baseline ==="
  python -m src.sft.train_sft
  purge_mps
fi

if [ "$SKIP_REWARD" = true ]; then
  skip_note "Part 2a: Reward Model (using checkpoint: checkpoints/reward_model/final)"
else
  echo "=== Part 2a: Reward Model ==="
  python -m src.rlhf.train_reward_model
  purge_mps
fi

if [ "$SKIP_RLHF" = true ]; then
  skip_note "Part 2b: RLHF PPO (using checkpoint: checkpoints/rlhf_ppo/final)"
else
  echo "=== Part 2b: RLHF PPO ==="
  python -m src.rlhf.train_ppo_rlhf
  purge_mps
fi

if [ "$SKIP_RLAIF" = true ]; then
  skip_note "Part 3: RLAIF PPO (using checkpoint: checkpoints/rlaif_ppo/final)"
else
  echo "=== Part 3: RLAIF PPO ==="
  python -m src.rlaif.train_ppo_rlaif
  purge_mps
fi

if [ "$SKIP_EVAL" = true ]; then
  skip_note "Evaluation"
else
  echo "=== Evaluation ==="
  python -m src.evaluation.evaluate
  python -m src.evaluation.reward_hacking
fi

echo "=== Done — results in results/ ==="
