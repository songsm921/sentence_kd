import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup
)
import numpy as np
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Knowledge Distillation for Llama Models")
    
    # Model configuration
    parser.add_argument("--teacher_model_name", type=str, default="meta-llama/Llama-2-8b-hf", 
                        help="Teacher model name or path")
    parser.add_argument("--student_model_name", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Student model name or path")
    
    # Training configuration
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Path to training data file")
    parser.add_argument("--output_dir", type=str, default="./distilled_model", 
                        help="Output directory for checkpoints and model")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Peak learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, 
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="Logging steps during training")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Steps interval to save model")
    
    # Distillation configuration
    parser.add_argument("--distillation_type", type=str, choices=["token", "sentence", "hybrid"], 
                        default="hybrid", help="Type of distillation to use")
    parser.add_argument("--initial_gate_value", type=float, default=0.5, 
                        help="Initial value for the gate in hybrid distillation")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature for softmax in token-level distillation")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    
    return parser.parse_args()


class TextDataset(Dataset):
    """Dataset for training the knowledge distillation model"""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Args:
            file_path: Path to the text file with training examples
            tokenizer: Tokenizer for encoding the text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # For causal LM, the labels are the same as input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }


class HybridDistillationModel(nn.Module):
    """
    Model that implements the hybrid distillation approach from the paper,
    combining token-level and sentence-level distillation with a learnable gate
    """
    
    def __init__(
        self, 
        student_model_name, 
        teacher_model_name, 
        distillation_type="hybrid",
        initial_gate_value=0.5,
        temperature=1.0
    ):
        """
        Args:
            student_model_name: Name or path of the student model
            teacher_model_name: Name or path of the teacher model
            distillation_type: Type of distillation ("token", "sentence", or "hybrid")
            initial_gate_value: Initial value for the gate parameter in hybrid distillation
            temperature: Temperature for softening the logits in token-level distillation
        """
        super().__init__()
        
        logger.info(f"Initializing student model from {student_model_name}")
        self.student = LlamaForCausalLM.from_pretrained(student_model_name)
        
        logger.info(f"Initializing teacher model from {teacher_model_name}")
        self.teacher = LlamaForCausalLM.from_pretrained(teacher_model_name)
        
        # Freeze the teacher model parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.distillation_type = distillation_type
        self.temperature = temperature
        
        # If using hybrid distillation, create a learnable gate parameter
        # This corresponds to the g(x) function in the paper (Section 4.1)
        if distillation_type == "hybrid":
            # Initialize the gate as a parameter that can be learned during training
            # Using sigmoid to ensure it stays between 0 and 1
            self.gate_logit = nn.Parameter(torch.tensor(
                np.log(initial_gate_value / (1 - initial_gate_value))
            ))
            logger.info(f"Initialized gate parameter with logit value: {self.gate_logit.item()}")
        else:
            self.gate_logit = None
            
    def get_gate_value(self):
        """Calculate the gate value using the sigmoid function as in Equation (1)"""
        if self.gate_logit is not None:
            return torch.sigmoid(self.gate_logit)
        return None
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        labels=None,
        return_dict=True
    ):
        """
        Forward pass with distillation loss calculation
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (same as input_ids for causal LM)
            return_dict: Whether to return a dictionary of outputs
            
        Returns:
            Dictionary containing loss and logits
        """
        # Get student model outputs
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        student_logits = student_outputs.logits
        
        # In inference mode (no labels provided), just return student outputs
        if labels is None:
            return student_outputs
            
        # Calculate standard CE loss from student outputs
        student_ce_loss = student_outputs.loss
        
        # Get teacher model outputs (without computing gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            teacher_logits = teacher_outputs.logits
            
        # Calculate the distillation loss based on the chosen method
        if self.distillation_type == "token":
            # Token-level distillation loss (Equation 2 in the paper)
            distillation_loss = self.compute_token_level_loss(
                student_logits, 
                teacher_logits, 
                attention_mask
            )
            total_loss = distillation_loss
            
        elif self.distillation_type == "sentence":
            # Sentence-level distillation loss (Equation 3 in the paper)
            distillation_loss = self.compute_sentence_level_loss(
                student_logits, 
                teacher_logits, 
                input_ids, 
                attention_mask
            )
            total_loss = distillation_loss
            
        elif self.distillation_type == "hybrid":
            # Hybrid loss combining token-level and sentence-level (Equation 4)
            token_loss = self.compute_token_level_loss(
                student_logits, 
                teacher_logits, 
                attention_mask
            )
            sentence_loss = self.compute_sentence_level_loss(
                student_logits, 
                teacher_logits, 
                input_ids, 
                attention_mask
            )
            
            # Use the gate parameter to combine the losses
            gate_value = self.get_gate_value()
            total_loss = gate_value * token_loss + (1 - gate_value) * sentence_loss
            
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                logger.warning(f"NaN or Inf in loss: token_loss={token_loss}, sentence_loss={sentence_loss}, gate={gate_value}")
                # Fall back to CE loss if distillation loss has numerical issues
                total_loss = student_ce_loss
        else:
            # Default to CE loss if no distillation method is specified
            total_loss = student_ce_loss
        
        return {
            "loss": total_loss,
            "logits": student_logits,
            "student_ce_loss": student_ce_loss
        }
    
    def compute_token_level_loss(self, student_logits, teacher_logits, attention_mask):
        """
        Compute token-level distillation loss as defined in Equation (2)
        
        This loss makes the student model match the token probability distribution
        of the teacher model at each position in the sequence.
        """
        vocab_size = student_logits.size(-1)
        
        # Apply temperature to soften the distributions
        soft_student_logits = student_logits / self.temperature
        soft_teacher_logits = teacher_logits / self.temperature
        
        # Apply softmax to get distributions
        student_probs = F.softmax(soft_student_logits, dim=-1)
        teacher_probs = F.softmax(soft_teacher_logits, dim=-1)
        
        # KL divergence between teacher and student distributions
        # Sum over the vocabulary dimension
        kl_div = F.kl_div(
            F.log_softmax(soft_student_logits, dim=-1),
            teacher_probs,
            reduction='none'
        ).sum(-1)
        
        # Apply attention mask to ignore padding tokens
        masked_kl_div = kl_div * attention_mask
        
        # Average over the sequence dimension and batch
        token_level_loss = masked_kl_div.sum() / attention_mask.sum()
        
        return token_level_loss
    
    def compute_sentence_level_loss(self, student_logits, teacher_logits, input_ids, attention_mask):
        """
        Compute sentence-level distillation loss as defined in Equation (3)
        
        This loss makes the student model produce the same output sequence
        as the teacher model would produce.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Get the most likely next tokens from the teacher (greedy)
        with torch.no_grad():
            teacher_preds = torch.argmax(teacher_logits, dim=-1)
        
        # Create label tensor where each position contains the teacher's prediction
        # for the next token (shifted right)
        # We'll use teacher's predictions as targets for all positions except the last one
        shifted_teacher_preds = torch.full((batch_size, seq_len), -100, device=device)
        shifted_teacher_preds[:, :-1] = teacher_preds[:, 1:].clone()
        
        # Apply attention mask to ensure we only compute loss on valid positions
        valid_positions = attention_mask.bool()
        
        # Calculate cross-entropy loss between student logits and teacher predictions
        sentence_level_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shifted_teacher_preds.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        return sentence_level_loss


def train_model(args):
    """
    Main training function for the distillation model
    """
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer from the student model (they should be compatible)
    tokenizer = LlamaTokenizer.from_pretrained(args.student_model_name)
    
    # Prepare dataset
    train_dataset = TextDataset(
        args.train_file, 
        tokenizer, 
        max_length=args.max_seq_length
    )
    
    # Create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize distillation model
    model = HybridDistillationModel(
        student_model_name=args.student_model_name,
        teacher_model_name=args.teacher_model_name,
        distillation_type=args.distillation_type,
        initial_gate_value=args.initial_gate_value,
        temperature=args.temperature
    )
    model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Calculate total steps
    total_steps = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Set up mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    model.train()
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        # Track losses for logging
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward and backward pass with automatic mixed precision
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    loss = loss / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                outputs = model(**batch)
                loss = outputs["loss"]
                loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # Update progress bar
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({
                "loss": loss.item() * args.gradient_accumulation_steps,
                "lr": scheduler.get_last_lr()[0]
            })
            
            # Log progress
            if global_step > 0 and global_step % args.logging_steps == 0:
                if args.distillation_type == "hybrid":
                    gate_value = model.get_gate_value().item()
                    logger.info(f"Step {global_step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.8f}, gate={gate_value:.4f}")
                else:
                    logger.info(f"Step {global_step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.8f}")
            
            # Save checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save only the student model
                model.student.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                # Save training arguments
                with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4)
                
                # Save gate value if using hybrid distillation
                if args.distillation_type == "hybrid":
                    gate_value = model.get_gate_value().item()
                    with open(os.path.join(output_dir, "gate_value.txt"), "w") as f:
                        f.write(f"{gate_value}")
                
                logger.info(f"Saved checkpoint to {output_dir}")
        
        # Log epoch stats
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save model after each epoch
        epoch_output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        # Save only the student model
        model.student.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        
        logger.info(f"Saved epoch checkpoint to {epoch_output_dir}")
    
    # Save the final model
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Save only the student model
    model.student.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save gate value if using hybrid distillation
    if args.distillation_type == "hybrid":
        gate_value = model.get_gate_value().item()
        with open(os.path.join(final_output_dir, "gate_value.txt"), "w") as f:
            f.write(f"{gate_value}")
    
    logger.info(f"Training completed. Final model saved to {final_output_dir}")
    
    return model


if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model = train_model(args)