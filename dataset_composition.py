import os
import argparse
import logging
import json
import random
import torch
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compose a dataset of math problems for LLM distillation")
    
    # Dataset selection
    parser.add_argument("--datasets", type=str, nargs="+", 
                        choices=["gsm8k", "open_platypus", "aime", "math"], 
                        default=["gsm8k"],
                        help="List of datasets to include in the composition")
    
    # Dataset specific configurations
    parser.add_argument("--gsm8k_split", type=str, default="train",
                        choices=["train", "test"],
                        help="Split of GSM8K dataset to use")
    
    parser.add_argument("--math_split", type=str, default="train",
                        choices=["train", "test"],
                        help="Split of MATH dataset to use")
                        
    parser.add_argument("--math_level", type=str, default="all",
                        choices=["all", "easy", "medium", "hard", "very_hard"],
                        help="Difficulty level for MATH dataset")
                        
    parser.add_argument("--aime_years", type=str, nargs="+", 
                        default=["all"],
                        help="Years to include from AIME (e.g., 2019 2020), or 'all'")
    
    # Output configuration
    parser.add_argument("--output_file", type=str, default="math_dataset.json",
                        help="Output file for the composed dataset")
    
    parser.add_argument("--train_file", type=str, default="train_math.txt",
                        help="Output file for the training data in the format required by knowledge_distillation.py")
    
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for training examples")
    
    parser.add_argument("--sample_size", type=int, default=3,
                        help="Number of samples to show from each dataset")
                        
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to include from each dataset (for testing)")
                        
    parser.add_argument("--train_test_split", type=float, default=0.9,
                        help="Ratio of the dataset to use for training (the rest will be used for testing)")
                        
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer to use for formatting the data")
                        
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def load_gsm8k(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """
    Load GSM8K dataset
    
    Args:
        split: Dataset split to use ('train' or 'test')
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        HuggingFace Dataset with standardized format
    """
    logger.info(f"Loading GSM8K dataset ({split} split)")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("gsm8k", "main", split=split)
    
    # Apply sampling if needed
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    logger.info(f"Loaded {len(dataset)} examples from GSM8K")
    
    # Standardize format
    def standardize_gsm8k(example):
        # Format problem and solution as expected for instruction tuning/distillation
        problem = f"Solve this math problem step by step:\n{example['question']}"
        solution = example['answer']
        
        return {
            "problem": problem,
            "solution": solution,
            "source": "gsm8k",
            "metadata": {
                "split": split
            }
        }
    
    # Apply the transformation
    standardized_dataset = dataset.map(standardize_gsm8k)
    
    return standardized_dataset

def load_open_platypus(max_samples: Optional[int] = None) -> Dataset:
    """
    Load Open-Platypus dataset (math subset)
    
    Args:
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        HuggingFace Dataset with standardized format
    """
    logger.info("Loading Open-Platypus dataset (math subset)")
    
    # Load from HuggingFace datasets
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
    
    # Filter for math problems
    # This is a simplification - we'd need to refine this filter for actual use
    math_keywords = ["math", "algebra", "calculus", "geometry", "arithmetic", 
                     "equation", "problem", "solve", "computation"]
    
    def is_math_problem(example):
        # Check if any math keyword appears in the input
        return any(keyword in example["input"].lower() for keyword in math_keywords)
    
    math_dataset = dataset.filter(is_math_problem)
    logger.info(f"Filtered {len(math_dataset)} math examples from Open-Platypus")
    
    # Apply sampling if needed
    if max_samples is not None and max_samples < len(math_dataset):
        math_dataset = math_dataset.select(range(max_samples))
    
    # Standardize format
    def standardize_platypus(example):
        # Platypus format is already well-formed for instruction tuning
        return {
            "problem": example["input"],
            "solution": example["output"],
            "source": "open_platypus",
            "metadata": {
                "split": "train"
            }
        }
    
    # Apply the transformation
    standardized_dataset = math_dataset.map(standardize_platypus)
    
    return standardized_dataset

def load_aime(years: List[str] = ["all"], max_samples: Optional[int] = None) -> Dataset:
    """
    Load AIME dataset
    
    Args:
        years: List of years to include, or ["all"] for all years
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        HuggingFace Dataset with standardized format
    """
    logger.info(f"Loading AIME dataset (years: {', '.join(years) if years != ['all'] else 'all'})")
    
    # For AIME, we might need to load a custom JSON file
    # This is a placeholder implementation
    try:
        dataset = load_dataset("competition_math", split="train")
        
        # Filter for AIME problems
        def is_aime(example):
            return example["source"] == "AIME"
        
        aime_dataset = dataset.filter(is_aime)
        
        # Filter by years if specified
        if years != ["all"]:
            def in_selected_years(example):
                example_year = example.get("year", "")
                if isinstance(example_year, str):
                    example_year = example_year.strip()
                return str(example_year) in years
            
            aime_dataset = aime_dataset.filter(in_selected_years)
        
        logger.info(f"Filtered {len(aime_dataset)} AIME examples")
        
        # Apply sampling if needed
        if max_samples is not None and max_samples < len(aime_dataset):
            aime_dataset = aime_dataset.select(range(max_samples))
        
        # Standardize format
        def standardize_aime(example):
            problem = f"Solve this AIME competition problem:\n{example['problem']}"
            return {
                "problem": problem,
                "solution": example["solution"],
                "source": "aime",
                "metadata": {
                    "year": example.get("year", ""),
                    "level": "competition"
                }
            }
        
        # Apply the transformation
        standardized_dataset = aime_dataset.map(standardize_aime)
        
        return standardized_dataset
        
    except Exception as e:
        logger.warning(f"Error loading AIME dataset: {e}")
        logger.warning("Creating a placeholder AIME dataset")
        
        # Create placeholder data
        aime_data = {
            "problem": [
                "Solve this AIME competition problem:\nFind the number of ordered pairs (m,n) of positive integers such that m and n are relatively prime and m/n = 3/8.",
                "Solve this AIME competition problem:\nThe sum of the coefficients in the expansion of (1 + x + x²)^(2022) is equal to m. What is the remainder when m is divided by 1000?",
                "Solve this AIME competition problem:\nFind the smallest positive integer n such that n² - 17n + 72 is a perfect square."
            ],
            "solution": [
                "Since m/n = 3/8, we have m = 3k and n = 8k for some rational number k. For m and n to be integers, k must be a rational number with denominator dividing both 3 and 8. Since 3 and 8 are coprime, k must be an integer. For m and n to be relatively prime, k = 1 is the only possibility. So the only ordered pair is (3,8). The answer is 1.",
                "The sum of coefficients is found by evaluating the polynomial at x = 1. So we need (1 + 1 + 1)^2022 = 3^2022. To find 3^2022 mod 1000, we can use Euler's theorem. The answer is 976.",
                "We have n² - 17n + 72 = k² for some integer k. Solving for n, we get n = (17 ± √(289 - 4k²))/2. For this to be an integer, 289 - 4k² must be a perfect square. Testing values, we find k = 4 works, giving n = 13. The answer is 13."
            ],
            "source": ["aime", "aime", "aime"],
            "metadata": [
                {"year": "2019", "level": "competition"},
                {"year": "2020", "level": "competition"},
                {"year": "2018", "level": "competition"}
            ]
        }
        
        # Create a dataset from the placeholder data
        aime_dataset = Dataset.from_dict(aime_data)
        
        # Apply sampling if needed
        if max_samples is not None and max_samples < len(aime_dataset):
            aime_dataset = aime_dataset.select(range(max_samples))
            
        return aime_dataset

def load_math_dataset(split: str = "train", level: str = "all", max_samples: Optional[int] = None) -> Dataset:
    """
    Load MATH dataset
    
    Args:
        split: Dataset split to use ('train' or 'test')
        level: Difficulty level ('all', 'easy', 'medium', 'hard', 'very_hard')
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        HuggingFace Dataset with standardized format
    """
    logger.info(f"Loading MATH dataset ({split} split, {level} level)")
    
    try:
        # Load from HuggingFace datasets
        dataset = load_dataset("hendrycks/math", split=split)
        
        # Filter by level if specified
        if level != "all":
            def is_level(example):
                return example["level"] == level
                
            dataset = dataset.filter(is_level)
            
        logger.info(f"Loaded {len(dataset)} examples from MATH dataset")
        
        # Apply sampling if needed
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        # Standardize format
        def standardize_math(example):
            problem_type = example.get("type", "")
            problem = f"Solve this {problem_type} problem:\n{example['problem']}"
            
            return {
                "problem": problem,
                "solution": example["solution"],
                "source": "math",
                "metadata": {
                    "level": example["level"],
                    "type": example["type"],
                    "split": split
                }
            }
        
        # Apply the transformation
        standardized_dataset = dataset.map(standardize_math)
        
        return standardized_dataset
        
    except Exception as e:
        logger.warning(f"Error loading MATH dataset: {e}")
        logger.warning("Creating a placeholder MATH dataset")
        
        # Create placeholder data
        math_data = {
            "problem": [
                "Solve this algebra problem:\nFind all solutions to the equation x² - 6x + 8 = 0.",
                "Solve this calculus problem:\nEvaluate the indefinite integral ∫ (3x² + 2x - 5) dx.",
                "Solve this geometry problem:\nA circle has radius 5. What is its area in terms of π?"
            ],
            "solution": [
                "x² - 6x + 8 = 0\n(x - 4)(x - 2) = 0\nx = 4 or x = 2",
                "∫ (3x² + 2x - 5) dx = 3x³/3 + 2x²/2 - 5x + C = x³ + x² - 5x + C",
                "A = πr² = π(5)² = 25π"
            ],
            "source": ["math", "math", "math"],
            "metadata": [
                {"level": "easy", "type": "algebra", "split": split},
                {"level": "medium", "type": "calculus", "split": split},
                {"level": "easy", "type": "geometry", "split": split}
            ]
        }
        
        # Create a dataset from the placeholder data
        math_dataset = Dataset.from_dict(math_data)
        
        # Apply sampling if needed
        if max_samples is not None and max_samples < len(math_dataset):
            math_dataset = math_dataset.select(range(max_samples))
            
        return math_dataset

def print_dataset_item(item: Dict[str, Any], index: int = None) -> None:
    """
    Print a formatted representation of a dataset item
    
    Args:
        item: The dataset item to print
        index: Optional index to display
    """
    index_str = f"[{index}] " if index is not None else ""
    source_str = f"Source: {item['source']}"
    if 'metadata' in item and item['metadata']:
        metadata_str = ", ".join(f"{k}: {v}" for k, v in item['metadata'].items())
        source_str += f" ({metadata_str})"
    
    print(f"\n{index_str}{source_str}")
    print("=" * 80)
    print("PROBLEM:")
    print(item['problem'])
    print("-" * 80)
    print("SOLUTION:")
    print(item['solution'])
    print("=" * 80)

def format_for_distillation(item: Dict[str, Any]) -> str:
    """
    Format a dataset item for the knowledge distillation training file
    
    Args:
        item: The dataset item to format
        
    Returns:
        A formatted string for the training file
    """
    # Format as an instruction-following example
    return f"{item['problem']}\n\n{item['solution']}"

def compose_dataset(args):
    """
    Compose a dataset from the specified sources
    
    Args:
        args: Command-line arguments
        
    Returns:
        Combined dataset
    """
    random.seed(args.seed)
    datasets = []
    
    # Load selected datasets
    if "gsm8k" in args.datasets:
        gsm8k_dataset = load_gsm8k(split=args.gsm8k_split, max_samples=args.max_samples)
        datasets.append(gsm8k_dataset)
        
    if "open_platypus" in args.datasets:
        platypus_dataset = load_open_platypus(max_samples=args.max_samples)
        datasets.append(platypus_dataset)
        
    if "aime" in args.datasets:
        aime_dataset = load_aime(years=args.aime_years, max_samples=args.max_samples)
        datasets.append(aime_dataset)
        
    if "math" in args.datasets:
        math_dataset = load_math_dataset(
            split=args.math_split, 
            level=args.math_level,
            max_samples=args.max_samples
        )
        datasets.append(math_dataset)
    
    # Combine datasets
    if not datasets:
        logger.error("No datasets selected! Please specify at least one dataset.")
        return None
        
    combined_dataset = concatenate_datasets(datasets)
    logger.info(f"Combined dataset has {len(combined_dataset)} examples")
    
    # Print sample items
    if args.sample_size > 0:
        print("\n" + "=" * 40 + " DATASET SAMPLES " + "=" * 40)
        for source in args.datasets:
            print(f"\n## Samples from {source.upper()} ##")
            
            # Filter for the current source
            source_indices = [
                i for i, item in enumerate(combined_dataset) 
                if item["source"] == source
            ]
            
            # Show random samples
            sample_indices = random.sample(
                source_indices, 
                min(args.sample_size, len(source_indices))
            )
            
            for idx in sample_indices:
                print_dataset_item(combined_dataset[idx], idx)
    
    return combined_dataset

def save_dataset(dataset, output_file):
    """
    Save the dataset to a file
    
    Args:
        dataset: The dataset to save
        output_file: Output file path
    """
    # Convert to a list of dictionaries
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Dataset saved to {output_file}")

def create_distillation_files(dataset, tokenizer, args):
    """
    Create files for the knowledge distillation process
    
    Args:
        dataset: The combined dataset
        tokenizer: The tokenizer to use for checking lengths
        args: Command-line arguments with configuration
    """
    # Convert dataset to list
    data_list = [dataset[i] for i in range(len(dataset))]
    
    # Shuffle data
    random.shuffle(data_list)
    
    # Split into train and test
    split_idx = int(len(data_list) * args.train_test_split)
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]
    
    logger.info(f"Split dataset into {len(train_data)} training examples and {len(test_data)} test examples")
    
    # Format training data for distillation
    train_formatted = []
    skipped = 0
    
    for item in train_data:
        formatted = format_for_distillation(item)
        
        # Check if the example fits within the max sequence length
        tokens = tokenizer(formatted, truncation=False, padding=False)
        if len(tokens["input_ids"]) <= args.max_seq_length:
            train_formatted.append(formatted)
        else:
            skipped += 1
    
    logger.info(f"Formatted {len(train_formatted)} training examples (skipped {skipped} that exceeded max length)")
    
    # Save training file (one example per line, as expected by TextDataset in knowledge_distillation.py)
    with open(args.train_file, 'w', encoding='utf-8') as f:
        for example in train_formatted:
            f.write(example + "\n\n")  # Double newline to separate examples
    
    logger.info(f"Saved training data to {args.train_file}")
    
    # Save a few test examples for evaluation
    test_file = args.train_file.replace(".txt", "_test.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data[:min(100, len(test_data))], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved test data to {test_file}")
    
    return train_formatted, test_data

def main():
    args = parse_args()
    
    logger.info(f"Composing dataset with the following sources: {', '.join(args.datasets)}")
    
    # Load tokenizer for sequence length checking
    try:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_name)
        logger.info(f"Loaded tokenizer from {args.tokenizer_name}")
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        logger.warning("Proceeding without tokenizer - won't check sequence lengths")
        tokenizer = None
    
    # Compose the dataset
    combined_dataset = compose_dataset(args)
    
    if combined_dataset is not None:
        # Save the raw dataset
        save_dataset(combined_dataset, args.output_file)
        
        # Create files for distillation training
        if tokenizer is not None:
            create_distillation_files(combined_dataset, tokenizer, args)
        else:
            logger.warning("Skipping distillation file creation due to missing tokenizer")

if __name__ == "__main__":
    main()