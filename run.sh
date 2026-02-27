#!/usr/bin/env bash
set -euo pipefail

# ── Single-GPU smoke test ──
# python train.py --max_steps 5 --output_dir ./output-smoke

# ── Single-GPU full run ──
# python train.py \
#     --output_dir ./output \
#     --run_name dapo-grpo-qwen3-1.7b

# ── Multi-GPU with accelerate ──
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir ./output \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_completion_length 1360 \
    --max_prompt_length 256 \
    --loss_type dr_grpo \
    --beta 0.0 \
    --temperature 1.0 \
    --num_iterations 1 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-15 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --save_steps 100 \
    --eval_steps 100 \
    --bf16 \
    --run_name dapo-grpo-qwen3-1.7b
