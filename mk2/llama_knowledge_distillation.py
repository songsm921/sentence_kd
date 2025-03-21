#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Distillation for Llama Models

This script implements three different knowledge distillation (KD) approaches:
1. Token-level KD: Student learns the token distribution of the teacher
2. Sentence-level KD: Student learns to match the output of the teacher
3. Hybrid KD: Combines both approaches with a learnable gating mechanism

The implementation supports Llama series models, uses the Open-platyplus 
dataset for math problems, and leverages DeepSpeed for distributed training.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed
)
from datasets import load_dataset
import deepspeed
import wandb

# Import config
from config import parse_args

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

#################################
# DATASET IMPLEMENTATION
#################################

class MathProblemDataset(Dataset):
    """Dataset class for Open-platyplus math problems with Chain-of-Thought (CoT) solutions"""
    
    def __init__(self, 
                 data, 
                 tokenizer,
                 max_seq_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as problem + solution (CoT format)
        problem = item["problem"]
        solution = item["solution"]
        
        # Combine as instruction format
        text = f"Problem: {problem}\n\nSolution: {solution}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract inputs and create labels (for causal LM, labels are the same as inputs)
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        labels = input_ids.clone()
        
        # Mask out padding tokens in labels (-100 is ignored in loss calculation)
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def prepare_datasets(tokenizer, max_seq_length: int):
    """Load and prepare the Open-platyplus dataset"""
    logger.info("Loading Open-platyplus dataset...")
    
    # Load dataset
    dataset = load_dataset("openplatypus/openplatyplus")
    
    # Filter for math problems only (optional)
    math_keywords = ["mathematics", "math", "algebra", "calculus", "geometry", "arithmetic"]
    
    # Filter for math problems
    math_data = []
    for item in dataset["train"]:
        for keyword in math_keywords:
            if keyword.lower() in item["problem"].lower():
                math_data.append(item)
                break
    
    # Create 90/10 train-eval split
    train_size = int(0.9 * len(math_data))
    train_data = math_data[:train_size]
    eval_data = math_data[train_size:]
    
    logger.info(f"Created dataset with {len(train_data)} training and {len(eval_data)} evaluation examples")
    
    # Create datasets
    train_dataset = MathProblemDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    
    eval_dataset = MathProblemDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    
    return train_dataset, eval_dataset

def create_dataloaders(train_dataset, eval_dataset, args, tokenizer):
    """Create DataLoaders for training and evaluation"""
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We want causal language modeling, not masked
    )
    
    # Create training dataloader
    if args.local_rank == -1:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4,
    )
    
    # Create evaluation dataloader
    eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=2,
    )
    
    return train_dataloader, eval_dataloader

#################################
# KNOWLEDGE DISTILLATION MODELS
#################################

class TokenLevelKD(nn.Module):
    """
    Token-level Knowledge Distillation
    Student model learns the token distribution of the teacher model.
    """
    
    def __init__(self, student_model, teacher_model, temperature=1.0):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        student_logits = student_outputs.logits
        
        # Get teacher outputs (no gradient tracking needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            teacher_logits = teacher_outputs.logits
        
        # Calculate token-level KD loss (KL divergence between distributions)
        # This is the implementation of the token-level distillation described in the paper
        # Student learns the output distribution of the teacher model at each token position
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
            log_target=False,
        ) * (self.temperature ** 2)
        
        # Combined loss (CE from labels + KD loss)
        loss = student_outputs.loss + kd_loss
        
        return {
            "loss": loss,
            "kd_loss": kd_loss,
            "ce_loss": student_outputs.loss,
            "logits": student_logits,
        }

class SentenceLevelKD(nn.Module):
    """
    Sentence-level Knowledge Distillation
    Student model learns to generate the same output as the teacher model.
    """
    
    def __init__(self, student_model, teacher_model):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Generate pseudo-targets from teacher
        with torch.no_grad():
            # For sentence-level KD, we use the teacher model to generate outputs
            # These outputs are then used as targets for the student model
            teacher_outputs = self.teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1],
                do_sample=False,
                num_beams=1,
            )
        
        # Get student outputs (using teacher outputs as labels)
        # This implements the sentence-level distillation from the paper
        # The student is trained with the teacher's output as the target
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=teacher_outputs,
            return_dict=True,
        )
        
        # Loss is already the cross-entropy between student predictions and teacher outputs
        loss = student_outputs.loss
        
        return {
            "loss": loss,
            "logits": student_outputs.logits,
        }

class HybridKD(nn.Module):
    """
    Hybrid Knowledge Distillation
    Combines token-level and sentence-level KD with a learnable gating mechanism.
    This implements the hybrid method described in the paper.
    """
    
    def __init__(self, student_model, teacher_model, temperature=1.0, initial_gate_value=0.5):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.temperature = temperature
        
        # Initialize the gate parameter as described in the paper
        # This will be learned during training to balance token-level and sentence-level KD
        self.gate = nn.Parameter(torch.tensor(initial_gate_value).log())
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def get_gate_value(self):
        """Convert gate parameter to a value between 0 and 1 using sigmoid"""
        return torch.sigmoid(self.gate)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        student_logits = student_outputs.logits
        
        with torch.no_grad():
            # Get teacher logits for token-level KD
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            teacher_logits = teacher_outputs.logits
            
            # Generate pseudo-targets from teacher for sentence-level KD
            teacher_generated = self.teacher_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_length=input_ids.shape[1],
                do_sample=False,
                num_beams=1,
            )
        
        # Calculate token-level KD loss (distribution matching)
        token_kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
            log_target=False,
        ) * (self.temperature ** 2)
        
        # Calculate sentence-level KD loss (matching teacher's generated outputs)
        # Calculate cross-entropy loss between student predictions and teacher's generated tokens
        sentence_kd_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_generated.view(-1),
            ignore_index=-100,
        )
        
        # Apply gating mechanism
        gate_value = self.get_gate_value()
        
        # Combine losses with gating
        # This implements the equation: L = g(x) * Ltoken-level + (1-g(x)) * Lsentence-level
        # From the paper's hybrid method
        kd_loss = gate_value * token_kd_loss + (1 - gate_value) * sentence_kd_loss
        
        # Final loss combines KD loss with standard CE loss
        loss = student_outputs.loss + kd_loss
        
        return {
            "loss": loss,
            "ce_loss": student_outputs.loss,
            "kd_loss": kd_loss,
            "token_kd_loss": token_kd_loss,
            "sentence_kd_loss": sentence_kd_loss,
            "gate_value": gate_value.item(),
            "logits": student_logits,
        }

def create_kd_model(kd_mode, student_model, teacher_model, temperature=1.0, initial_gate_value=0.5):
    """Create the appropriate KD model based on the mode"""
    if kd_mode == "token":
        return TokenLevelKD(student_model, teacher_model, temperature)
    elif kd_mode == "sentence":
        return SentenceLevelKD(student_model, teacher_model)
    elif kd_mode == "hybrid":
        return HybridKD(student_model, teacher_model, temperature, initial_gate_value)
    else:
        raise ValueError(f"Unknown KD mode: {kd_mode}")

#################################
# TRAINING FUNCTIONS
#################################

def train(args):
    """Main training function"""
    # Set up distributed training
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb and (args.local_rank == -1 or args.local_rank == 0):
        wandb_name = args.wandb_name or f"{args.kd_mode}-{args.teacher_model_name_or_path.split('/')[-1]}-{args.student_model_name_or_path.split('/')[-1]}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
        )
    
    # Create output directory
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save arguments
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.teacher_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load teacher model
    logger.info(f"Loading teacher model from {args.teacher_model_name_or_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.local_rank == -1 else {"": args.local_rank},
    )
    
    # Load student model
    if args.from_checkpoint:
        logger.info(f"Loading student model from checkpoint {args.from_checkpoint}")
        student_model = AutoModelForCausalLM.from_pretrained(
            args.from_checkpoint,
            torch_dtype=torch.bfloat16,
        )
    else:
        logger.info(f"Loading student model from {args.student_model_name_or_path}")
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
    
    # Load datasets
    train_dataset, eval_dataset = prepare_datasets(tokenizer, args.max_seq_length)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, args, tokenizer)
    
    # Initialize KD model
    logger.info(f"Initializing {args.kd_mode} knowledge distillation")
    kd_model = create_kd_model(
        kd_mode=args.kd_mode,
        student_model=student_model,
        teacher_model=teacher_model,
        temperature=args.temperature,
        initial_gate_value=args.initial_gate_value,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if len(train_dataloader) % args.gradient_accumulation_steps != 0:
        num_update_steps_per_epoch += 1
    max_steps = num_update_steps_per_epoch * args.num_train_epochs
    
    # Initialize optimizer
    optimizer_params = [
        {
            "params": [p for n, p in kd_model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay,
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.learning_rate)
    
    # Initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_steps,
    )
    
    # Initialize DeepSpeed if requested
    if args.deepspeed:
        # Convert model to DeepSpeed model
        logger.info(f"Initializing DeepSpeed with config {args.deepspeed}")
        kd_model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=kd_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=args.deepspeed,
            dist_init_required=True,
        )
    elif args.local_rank == -1:
        # Single GPU training
        kd_model = kd_model.to(torch.device("cuda"))
    
    # Log information about the training
    logger.info("***** Training information *****")
    logger.info(f"  KD mode = {args.kd_mode}")
    logger.info(f"  Teacher model = {args.teacher_model_name_or_path}")
    logger.info(f"  Student model = {args.student_model_name_or_path}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(max_steps),
        disable=(args.local_rank != -1 and args.local_rank != 0),
        desc="Training",
    )
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
        
        # Set to training mode
        kd_model.train()
        
        # Reset dataloader for distributed training
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to the appropriate device
            if args.deepspeed:
                # DeepSpeed handles device placement
                pass
            else:
                batch = {k: v.to(torch.device("cuda")) for k, v in batch.items()}
            
            # Forward pass
            outputs = kd_model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            if args.deepspeed:
                # DeepSpeed handles loss scaling and backward
                kd_model.backward(loss)
                kd_model.step()
            else:
                # Manually handle gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                # Update parameters if we've accumulated enough gradients
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)
            
            # Log progress
            if global_step % args.logging_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                # Prepare log data
                log_data = {"step": global_step, "loss": loss.item() * args.gradient_accumulation_steps}
                
                # Add KD-specific metrics
                if args.kd_mode == "token":
                    log_data.update({
                        "kd_loss": outputs["kd_loss"].item(),
                        "ce_loss": outputs["ce_loss"].item(),
                    })
                elif args.kd_mode == "hybrid":
                    log_data.update({
                        "kd_loss": outputs["kd_loss"].item(),
                        "token_kd_loss": outputs["token_kd_loss"].item(),
                        "sentence_kd_loss": outputs["sentence_kd_loss"].item(),
                        "gate_value": outputs["gate_value"],
                        "ce_loss": outputs["ce_loss"].item(),
                    })
                
                # Add learning rate
                log_data["learning_rate"] = lr_scheduler.get_last_lr()[0]
                
                # Update progress bar
                progress_bar.set_postfix(**{k: v for k, v in log_data.items() if k != "step"})
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log(log_data)
            
            # Evaluate model
            if global_step % args.eval_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                logger.info("Evaluating model...")
                eval_results = evaluate(kd_model, eval_dataloader, args)
                
                # Log evaluation results
                if args.use_wandb:
                    wandb.log({"eval_" + k: v for k, v in eval_results.items()})
                
                # Print evaluation results
                logger.info(f"Evaluation results: {eval_results}")
                
                # Set back to training mode
                kd_model.train()
            
            # Save checkpoint
            if global_step % args.save_steps == 0 and global_step > 0 and (args.local_rank == -1 or args.local_rank == 0):
                save_checkpoint(kd_model, tokenizer, args, global_step)
            
            # Check if we've reached max steps
            if global_step >= max_steps:
                break
    
    # Save final model
    if args.local_rank == -1 or args.local_rank == 0:
        logger.info("Training complete, saving final model")
        save_checkpoint(kd_model, tokenizer, args, global_step, final=True)
    
    return global_step

def evaluate(kd_model, eval_dataloader, args):
    """Evaluate the model on the evaluation dataset"""
    # Set to evaluation mode
    kd_model.eval()
    
    total_loss = 0
    total_kd_loss = 0
    total_ce_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to the appropriate device
            if args.deepspeed:
                # DeepSpeed handles device placement
                pass
            else:
                batch = {k: v.to(torch.device("cuda")) for k, v in batch.items()}
            
            # Forward pass
            outputs = kd_model(**batch)
            
            # Accumulate losses
            total_loss += outputs["loss"].item()
            
            # Add KD-specific metrics if available
            if "kd_loss" in outputs:
                total_kd_loss += outputs["kd_loss"].item()
            if "ce_loss" in outputs:
                total_ce_loss += outputs["ce_loss"].item()
            
            total_steps += 1
    
    # Calculate average losses
    avg_loss = total_loss / total_steps
    result = {"loss": avg_loss}
    
    # Add KD-specific metrics if calculated
    if total_kd_loss > 0:
        result["kd_loss"] = total_kd_loss / total_steps
    if total_ce_loss > 0:
        result["ce_loss"] = total_ce_loss / total_steps
    
    # Add gate value for hybrid KD
    if args.kd_mode == "hybrid":
        result["gate_value"] = kd_model.get_gate_value().item()
    
    return result

def save_checkpoint(kd_model, tokenizer, args, global_step, final=False):
    """Save model checkpoint"""
    # Determine output directory
    if final:
        output_dir = os.path.join(args.output_dir, "final")
    else:
        output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save student model
    if args.deepspeed:
        # Using the DeepSpeed checkpoint saving utility
        kd_model.save_checkpoint(output_dir)
        
        # Also save the model in HuggingFace format
        if hasattr(kd_model, "student_model"):
            kd_model.student_model.save_pretrained(output_dir)
    else:
        # Regular saving
        if hasattr(kd_model, "student_model"):
            kd_model.student_model.save_pretrained(output_dir)
        else:
            kd_model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training arguments
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Saved checkpoint to {output_dir}")
    
    # Clean up old checkpoints if needed
    if args.save_total_limit > 0:
        _cleanup_checkpoints(args)

def _cleanup_checkpoints(args):
    """Remove old checkpoints if we exceed the save_total_limit"""
    if args.save_total_limit <= 0:
        return
    
    checkpoint_dirs = [
        d for d in os.listdir(args.output_dir) 
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
    ]
    
    if len(checkpoint_dirs) <= args.save_total_limit:
        return
    
    # Sort by step number
    checkpoint_steps = [int(d.split("-")[1]) for d in checkpoint_dirs]
    checkpoint_dirs = [d for _, d in sorted(zip(checkpoint_steps, checkpoint_dirs))]
    
    # Remove the oldest checkpoints
    for d in checkpoint_dirs[:-args.save_total_limit]:
        dir_path = os.path.join(args.output_dir, d)
        logger.info(f"Removing old checkpoint {dir_path}")
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        try:
            os.rmdir(dir_path)
        except Exception as e:
            logger.warning(f"Failed to delete directory {dir_path}: {e}")

#################################
# MAIN FUNCTION
#################################

def main():
    """Main function to run the script"""
    # Parse arguments using imported function
    args = parse_args()
    
    # Start training
    train(args)
    
    # Clean up
    if args.use_wandb and (args.local_rank == -1 or args.local_rank == 0):
        wandb.finish()

if __name__ == "__main__":
    main()