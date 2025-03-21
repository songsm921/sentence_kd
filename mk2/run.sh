#!/bin/bash

# Define variables
TEACHER_MODEL="meta-llama/Llama-3.1-8B"
STUDENT_MODEL="meta-llama/Llama-3.2-3B"
OUTPUT_DIR="./output"
KD_MODE="hybrid"  # Options: token, sentence, hybrid

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with DeepSpeed on 8 GPUs
deepspeed --num_gpus=8 llama_knowledge_distillation.py \
  --teacher_model_name_or_path $TEACHER_MODEL \
  --student_model_name_or_path $STUDENT_MODEL \
  --output_dir $OUTPUT_DIR/$KD_MODE \
  --kd_mode $KD_MODE \
  --initial_gate_value 0.5 \
  --temperature 1.0 \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --warmup_steps 500 \
  --logging_steps 10 \
  --save_steps 1000 \
  --eval_steps 500 \
  --deepspeed ds_config.json \
  --use_wandb \
  --wandb_project "llama-kd" \
  --wandb_name "${KD_MODE}-distillation"