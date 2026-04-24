#!/bin/bash
set -eo pipefail

source "$(dirname "$0")/../.venv/bin/activate"
echo "Python: $(which python)"

# Load API keys from .env
set -a; source "$(dirname "$0")/../.env"; set +a

touch logs/full_run_v7.log

echo "=== Starting RLAIF GRPO (v8: fixed ranking parser + conciseness prompt) ===" | tee -a logs/full_run_v7.log

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -u -m src.rlaif.train_ppo_rlaif 2>&1 | tee -a logs/full_run_v7.log

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python -u -m src.evaluation.evaluate 2>&1 | tee -a logs/full_run_v7.log

python -u -m src.evaluation.reward_hacking 2>&1 | tee -a logs/full_run_v7.log
