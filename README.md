# RLHF vs RLAIF

Comparing two LLM alignment approaches on GPT-2 medium using the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset:

- **RLHF** — fine-tune a reward model on human preference pairs, then use GRPO to optimize the policy against it
- **RLAIF** — skip the reward model entirely; use Claude as a real-time judge during GRPO training

Both are compared against an SFT baseline. Evaluation scores each model with Claude and breaks results down by prompt category (helpful vs safety-sensitive).

## Pipeline

```
SFT baseline  →  reward model  →  RLHF policy
                                                  →  evaluation + reward hacking analysis
SFT baseline  →  RLAIF policy  ──────────────────┘
```

| Stage | Script | Output |
|-------|--------|--------|
| 1. SFT | `src/sft/train_sft.py` | `checkpoints/sft/final` |
| 2. Reward model | `src/rlhf/train_reward_model.py` | `checkpoints/reward_model/final` |
| 3. RLHF GRPO | `src/rlhf/train_ppo_rlhf.py` | `checkpoints/rlhf_ppo/final` |
| 4. RLAIF GRPO | `src/rlaif/train_ppo_rlaif.py` | `checkpoints/rlaif_ppo/final` |
| 5. Evaluation | `src/evaluation/evaluate.py` | `results/evaluation_results_{timestamp}.json` |
| 6. Reward hacking | `src/evaluation/reward_hacking.py` | `results/{label}_hacking_analysis_{timestamp}.json` |

## Setup

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
pip install -r requirements.txt
```

Requires an Anthropic API key for RLAIF training and evaluation.

## Running

```bash
# Full run
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 bash scripts/run_all.sh

# Individual stages (run from project root)
sudo purge
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.sft.train_sft
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlhf.train_reward_model
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlhf.train_ppo_rlhf
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.rlaif.train_ppo_rlaif
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.evaluation.evaluate
python -m src.evaluation.reward_hacking

# Smoke test (~10 min, 8 samples)
bash scripts/smoke_test.sh
```

## Design

- **Model**: GPT-2 medium (355M) — small enough to iterate quickly on a laptop
- **Data**: 5000 samples from HH-RLHF for SFT and reward model; 1000 prompts for GRPO
- **SFT**: LoRA rank 8, merged into base weights before saving; early stopping on eval loss
- **RLHF**: full fine-tune reward model (70% pairwise accuracy on held-out pairs); GRPO with tanh-normalized rewards
- **RLAIF**: Claude scores responses 1–10 on helpfulness/conciseness, normalized to [-1, 1] for GRPO
- **Evaluation**: Claude judge calibrated for small-model outputs; scores broken down by prompt category

## Hardware

Optimized for Apple M3 Pro (18GB unified memory, MPS backend). Run `sudo purge` before memory-intensive stages and close other applications. Expected runtimes:

| Stage | Time |
|-------|------|
| SFT | ~70 min |
| Reward model | ~90 min |
| RLHF GRPO | ~2.5 hr |
| RLAIF GRPO | ~3–4 hr |
| Evaluation | ~20 min |

## Dependencies

See `requirements.txt`. Key packages: `transformers`, `trl>=0.12`, `peft`, `anthropic`, `datasets`.

TRL 0.12+ API notes: `SFTConfig(max_length=)`, `GRPOConfig(max_completion_length=, beta=)`, `RewardTrainer` expects raw `{"chosen": str, "rejected": str}` — not pre-tokenized.
