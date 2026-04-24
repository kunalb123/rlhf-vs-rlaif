#!/bin/bash
set -e
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -u -m src.evaluation.evaluate 2>&1 | tee -a logs/full_run_v5.log
python -u -m src.evaluation.reward_hacking 2>&1 | tee -a logs/full_run_v5.log
