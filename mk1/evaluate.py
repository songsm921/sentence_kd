import os
import torch
import argparse
import logging
import json
import numpy as np
import sacrebleu
from tqdm import tqdm
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate distilled LLMs on translation tasks")
    
    # Model paths
    parser.add_argument("--teacher_model_path", type=str, required=True, 
                        help="Path to the teacher model")
    parser.add_argument("--student_model_path", type=str, required=True, 
                        help="Path to the student model (distilled)")
    parser.add_argument("--baseline_model_path", type=str, default=None, 
                        help="Path to a baseline model for comparison")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True, choices=["iwslt14", "iwslt13", "wmt14", "iwslt17"],
                        help="Dataset to evaluate on")
    parser.add_argument("--src_lang", type=str, required=True, 
                        help="Source language code (e.g., 'de', 'en', 'fr', 'ar')")
    parser.add_argument("--tgt_lang", type=str, required=True, 
                        help="Target language code (e.g., 'de', 'en', 'fr', 'ar')")
    parser.add_argument("--max_examples", type=int, default=1000, 
                        help="Maximum number of examples to evaluate")
    
    # Generation configuration
    parser.add_argument("--max_length", type=int, default=128, 
                        help="Maximum sequence length for generation")
    parser.add_argument("--beam_size", type=int, default=4, 
                        help="Beam size for beam search")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    
    # Output configuration
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", 
                        help="File to save evaluation results")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for evaluation")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use FP16 precision for faster evaluation")
    
    return parser.parse_args()


class TranslationDataset(Dataset):
    """
    Dataset for translation evaluation
    
    This dataset formats examples for translation evaluation,
    creating prompts with source text and preparing target references.
    """
    
    def __init__(self, examples, tokenizer, max_length=128, src_lang="en", tgt_lang="de"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        source = example['translation'][self.src_lang]
        target = example['translation'][self.tgt_lang]
        
        # Format the input for translation
        input_text = f"Translate from {self.src_lang} to {self.tgt_lang}: {source}"
        
        # Tokenize the input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "source": source,
            "target": target
        }


def load_evaluation_dataset(dataset_name, src_lang, tgt_lang, tokenizer, max_examples, max_length=128):
    """
    Load and prepare the evaluation dataset for translation tasks
    
    Args:
        dataset_name: Name of the dataset (iwslt14, iwslt13, wmt14, iwslt17)
        src_lang: Source language code
        tgt_lang: Target language code
        tokenizer: Tokenizer to use for preparing inputs
        max_examples: Maximum number of examples to use
        max_length: Maximum sequence length
        
    Returns:
        TranslationDataset instance
    """
    """Load the evaluation dataset based on the specified benchmark"""
    
    if dataset_name == "iwslt14":
        if (src_lang == "de" and tgt_lang == "en") or (src_lang == "en" and tgt_lang == "de"):
            # IWSLT 2014 German-English
            dataset = load_dataset("iwslt2017", f"{src_lang}-{tgt_lang}", split="test")
        else:
            raise ValueError(f"Language pair {src_lang}-{tgt_lang} not supported for IWSLT14")
            
    elif dataset_name == "iwslt13":
        if (src_lang == "en" and tgt_lang == "fr") or (src_lang == "fr" and tgt_lang == "en"):
            # IWSLT 2013 English-French
            dataset = load_dataset("iwslt2017", f"{src_lang}-{tgt_lang}", split="test")
        else:
            raise ValueError(f"Language pair {src_lang}-{tgt_lang} not supported for IWSLT13")
            
    elif dataset_name == "wmt14":
        if (src_lang == "en" and tgt_lang == "de") or (src_lang == "de" and tgt_lang == "en"):
            # WMT 2014 English-German
            dataset = load_dataset("wmt14", f"{src_lang}-{tgt_lang}", split="test")
        else:
            raise ValueError(f"Language pair {src_lang}-{tgt_lang} not supported for WMT14")
            
    elif dataset_name == "iwslt17":
        if (src_lang == "ar" and tgt_lang == "en") or (src_lang == "en" and tgt_lang == "ar"):
            # IWSLT 2017 Arabic-English
            dataset = load_dataset("iwslt2017", f"{src_lang}-{tgt_lang}", split="test")
        else:
            raise ValueError(f"Language pair {src_lang}-{tgt_lang} not supported for IWSLT17")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit the number of examples if specified
    if max_examples > 0 and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))
    
    logger.info(f"Loaded {len(dataset)} examples from {dataset_name} {src_lang}-{tgt_lang}")
    
    # Create a TranslationDataset instance
    translation_dataset = TranslationDataset(
        dataset,
        tokenizer,
        max_length=max_length,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    
    return translation_dataset


def calculate_bleu(hypotheses, references):
    """
    Calculate BLEU score for the translations
    
    Args:
        hypotheses: List of generated translations
        references: List of reference translations
        
    Returns:
        BLEU score (float)
    """
    # Handle empty translations by replacing with a placeholder
    clean_hypotheses = [h if h.strip() else "empty_translation" for h in hypotheses]
    
    # Compute BLEU score using sacrebleu
    bleu = sacrebleu.corpus_bleu(clean_hypotheses, [references])
    return bleu.score


def evaluate_model(model, tokenizer, dataloader, device, beam_size=4, max_length=128, fp16=False, model_name="model"):
    """
    Evaluate a model on a translation dataset
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        dataloader: DataLoader with evaluation examples
        device: Device to run the model on
        beam_size: Beam size for beam search
        max_length: Maximum length for generated sequences
        fp16: Whether to use mixed precision
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)
    
    hypotheses = []
    references = []
    
    logger.info(f"Starting evaluation of {model_name}...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate translations
        with torch.no_grad():
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=beam_size,
                        early_stopping=True
                    )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=beam_size,
                    early_stopping=True
                )
        
        # Decode the generated outputs
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract the actual translations (removing the prompt)
        translations = []
        for output in decoded_outputs:
            # Extract the translation from the output (after the prompt)
            translation_parts = output.split(":")
            if len(translation_parts) > 1:
                translation = translation_parts[-1].strip()
            else:
                translation = output  # Use the whole output if we can't find the delimiter
            translations.append(translation)
        
        # Add to lists for BLEU calculation
        hypotheses.extend(translations)
        references.extend(batch["target"])
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(hypotheses, references)
    
    logger.info(f"{model_name} BLEU score: {bleu_score:.2f}")
    
    return {
        "bleu": bleu_score,
        "num_examples": len(references),
        "translations": list(zip(references, hypotheses))[:10],  # Save first 10 translations as examples
        "model_name": model_name
    }