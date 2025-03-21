#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration and argument parsing utilities for Llama Knowledge Distillation
"""

import argparse
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for knowledge distillation training."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Llama Models")
    
    # KD settings
    parser.add_argument(
        "--kd_mode",
        type=str,
        default="hybrid",
        choices=["token", "sentence", "hybrid"],
        help="Knowledge distillation mode: token-level, sentence-level, or hybrid",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softening the teacher distribution in token-level KD",
    )
    parser.add_argument(
        "--initial_gate_value",
        type=float,
        default=0.5,
        help="Initial gate value (0-1) for hybrid KD mode",
    )
    
    # Model settings
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Student model name or path",
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="Load student model from checkpoint",
    )
    
    # Dataset settings
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    
    # Training settings
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluation steps",
    )
    
    # DeepSpeed settings
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    
    # Wandb settings
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama-knowledge-distillation",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    
    args = parser.parse_args()
    return args