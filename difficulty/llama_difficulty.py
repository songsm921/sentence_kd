import argparse
import concurrent.futures
import json
import os
import pandas as pd
from concurrent.futures import as_completed
from copy import deepcopy
from tqdm import tqdm

# Replace these imports with your actual module paths
from deepscaler.system_prompts import MATH_DIFFICULTY_PROMPT
# No longer using Gemini API
# from deepscaler.utils import call_gemini_llm

# Import libraries for Llama 3.1
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the dataset loading logic
from deepscaler.data.dataset_types import (
    TrainDataset,
    TestDataset,
)
from deepscaler.data.utils import load_dataset

# Global variables for Llama model (initialized once)
llama_model = None
llama_tokenizer = None

def init_llama_model(model_name="/mnt/cephfs/sumin/model/Llama-3.1-8B-Instruct", device="cuda"):
    """
    Initialize the Llama 3.1 model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        device: Device to load the model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global llama_model, llama_tokenizer
    
    if llama_model is None or llama_tokenizer is None:
        print(f"Loading Llama model: {model_name}")
        
        # Check if CUDA is available when device is "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            
        # Load tokenizer
        llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate precision based on device
        if device == "cuda":
            # Use BF16 precision for faster inference on compatible GPUs
            llama_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            # Use FP32 precision for CPU
            llama_model = AutoModelForCausalLM.from_pretrained(model_name)
            llama_model.to(device)
    
    return llama_model, llama_tokenizer

def call_llama_llm(prompt, system_prompt, n=1, temperature=0.7, max_tokens=512):
    """
    Call Llama 3.1 model for text generation
    
    Args:
        prompt: The input prompt
        system_prompt: System instruction for the model
        n: Number of generations to perform
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
    
    Returns:
        List of generated texts
    """
    model, tokenizer = init_llama_model()
    
    # Format the prompt for Llama 3.1
    formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
    
    results = []
    for _ in range(n):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate with sampling for diverse outputs
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the generation
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Add to results
        results.append(response.strip())
    
    return results

def difficulty_fn(idx, entry, dataset_type="standard"):
    """
    1) Extract problem and solution text.
    2) Call LLM for difficulty estimates (8 numeric strings).
    3) Convert to float safely, filter out parse errors.
    4) Take the average and store as 'difficulty'.
    
    Args:
        idx: Index of the entry
        entry: Dictionary containing problem data
        dataset_type: Type of dataset format ("standard" or "open_platyplus")
    """
    # Skip if already computed
    if entry.get('difficulty') is not None:
        return idx, entry

    # Extract problem and solution based on dataset type
    if dataset_type == "standard":
        problem_text = entry.get('problem', '')
        solution_text = entry.get('solution', '')
    elif dataset_type == "open_platyplus":
        problem_text = entry.get('instruction', '')
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')
        solution_text = f"Input: {input_text}\nOutput: {output_text}"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Pass@8 difficulty calls using Llama instead of Gemini
    output_list = call_llama_llm(
        f"Problem: {problem_text}\n----\nSolution: {solution_text}",
        system_prompt=MATH_DIFFICULTY_PROMPT,
        n=8,
        temperature=0.7,
    )
    # (Use .lower() to catch both uppercase/lowercase errors)
    output_list = [
        o for o in output_list
        if 'error' not in o.lower()
    ]

    # Attempt to parse each string as float
    values = []
    for o in output_list:
        try:
            val = float(o)
            values.append(val)
        except ValueError:
            # Ignore anything that can't be parsed as float
            pass

    # Compute the average or set None if no valid floats
    if values:
        difficulty = sum(values) / len(values)
    else:
        difficulty = None
        print('Failed parsing all difficulties: ', output_list)

    entry['difficulty'] = difficulty
    return idx, entry


def load_open_platyplus(file_path, split="train"):
    """
    Load Open Platyplus dataset from Parquet file
    
    Args:
        file_path: Path to the Parquet file
        split: Dataset split (train, test, validation)
        
    Returns:
        List of dictionaries containing the dataset entries
    """
    df = pd.read_parquet(file_path)
    data = df.to_dict('records')
    return data


def save_open_platyplus(data, output_path, partial=False, count=None):
    """
    Save Open Platyplus dataset to Parquet format
    
    Args:
        data: List of dictionaries to save
        output_path: Path where to save the file
        partial: Whether this is a partial save
        count: Number of entries if partial
    """
    df = pd.DataFrame(data)
    save_path = output_path
    if partial and count is not None:
        save_path = f"{output_path}_partial_{count}.parquet"
    df.to_parquet(save_path, index=False)
    return save_path


def batch_difficulty(dataset=None, split=None, dataset_type="standard", 
                     input_file=None, output_file=None):
    """
    Process a dataset to compute difficulty scores for each entry
    
    Args:
        dataset: Name of the standard dataset (for standard dataset type)
        split: Split to process (train or test)
        dataset_type: Type of dataset ("standard" or "open_platyplus")
        input_file: Input file path (for open_platyplus dataset)
        output_file: Output file path (for open_platyplus dataset)
    """
    if dataset_type == "standard":
        # Standard dataset processing (AIME, AMC, OLYMPIAD, etc.)
        if dataset is None or split is None:
            raise ValueError("For standard datasets, 'dataset' and 'split' must be provided")
            
        # Figure out if we need a TrainDataset or TestDataset
        if split == "train":
            dataset_enum = TrainDataset[dataset.upper()]
        else:
            dataset_enum = TestDataset[dataset.upper()]

        # Load data using the provided load_dataset function
        data = load_dataset(dataset_enum)
        results = deepcopy(data)

        # Prepare to save back to the same file location
        data_dir = "train_llama_ver" if isinstance(dataset_enum, TrainDataset) else "test"
        dataset_name = dataset_enum.value.lower()
        file_path = os.path.join("..", data_dir, f"{dataset_name}.json")
        
        # Define save function for standard datasets
        def save_partial(data, partial=False, count=None):
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return file_path
            
    elif dataset_type == "open_platyplus":
        # Open Platyplus dataset processing
        if input_file is None:
            raise ValueError("For open_platyplus dataset, 'input_file' must be provided")
            
        # Load the dataset
        data = load_open_platyplus(input_file, split)
        results = deepcopy(data)
        
        # Define file path for saving
        file_path = output_file if output_file else f"{input_file}_with_difficulty.parquet"
        
        # Use the specific save function for open_platyplus
        save_partial = save_open_platyplus
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Use ThreadPoolExecutor to process concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(difficulty_fn, i, entry, dataset_type)
            for i, entry in enumerate(data)
        ]
        done_count = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            done_count += 1

            # Periodically save partial results
            if done_count % 5000 == 0:
                print(f"Processed {done_count} entries so far. Saving partial results...")
                save_path = save_partial(results, partial=True, count=done_count)
                print(f"Partial results saved to {save_path}")
                
    # Save final results
    final_path = save_partial(results)
    print(f"Finished processing {len(results)} entries. Results saved to {final_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label approximate difficulty for math problems.")
    
    # Dataset type selection
    parser.add_argument(
        "--dataset_type",
        default="standard",
        choices=["standard", "open_platyplus"],
        help="Type of dataset to process: 'standard' (AIME, MATH, etc.) or 'open_platyplus'"
    )
    
    # Arguments for standard datasets
    parser.add_argument(
        "--dataset",
        help="Name of the standard dataset (e.g. 'AIME', 'AMC', 'OMNI_MATH', 'OLYMPIAD', 'MATH')"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation"],
        help="Which split to use: 'train', 'test', or 'validation'"
    )
    
    # Arguments for Open Platyplus dataset
    parser.add_argument(
        "--input_file",
        help="Path to the Open Platyplus Parquet file"
    )
    parser.add_argument(
        "--output_file",
        help="Path to save the output file with difficulty scores (for Open Platyplus)"
    )
    
    # Arguments for Llama model
    parser.add_argument(
        "--model_name",
        default="/mnt/cephfs/sumin/model/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path for Llama model"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run the model on (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for model inference"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on dataset type
    if args.dataset_type == "standard" and not args.dataset:
        parser.error("--dataset is required when --dataset_type is 'standard'")
    elif args.dataset_type == "open_platyplus" and not args.input_file:
        parser.error("--input_file is required when --dataset_type is 'open_platyplus'")
    
    # Initialize the Llama model
    init_llama_model(model_name=args.model_name, device=args.device)
    
    # Run the appropriate function based on dataset type
    batch_difficulty(
        dataset=args.dataset,
        split=args.split,
        dataset_type=args.dataset_type,
        input_file=args.input_file,
        output_file=args.output_file
    )