#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset utilities for Llama Knowledge Distillation
This module handles loading and preprocessing of the Open-platyplus dataset for math problems.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MathProblemDataset(Dataset):
    """
    Dataset class for Open-platyplus math problems with solutions
    
    The Open-platyplus dataset has the following structure:
    - instruction: the problem statement
    - input: (usually empty for math problems)
    - output: the solution/answer
    - data_source: source of the data (e.g., "MATH/PRM-800K")
    """
    
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
        
        # Extract fields from the dataset
        instruction = item["instruction"]  # The problem statement
        output = item["output"]           # The solution
        
        # Combine as instruction format (problem + solution)
        text = f"Problem: {instruction}\n\nSolution: {output}"
        
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
    """
    Load and prepare the Open-platyplus dataset
    
    Filters for math problems from MATH/PRM-800K source
    """
    logger.info("Loading Open-platyplus dataset...")
    
    # Load dataset
    dataset = load_dataset("garage-bAInd/Open-Platypus")
    
    # Use the entire dataset for math problems from MATH/PRM-800K
    math_data = dataset['train']
    
    # Filter for math problems if needed
    filtered_math_data = []
    for item in math_data:
        # if "MATH/PRM-800K" in item.get("data_source", ""):
        filtered_math_data.append(item)
    
    logger.info(f"Found {len(filtered_math_data)} math problems from MATH/PRM-800K")
    
    # Create 90/10 train-eval split
    train_size = int(0.9 * len(filtered_math_data))
    train_data = filtered_math_data[:train_size]
    eval_data = filtered_math_data[train_size:]
    
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