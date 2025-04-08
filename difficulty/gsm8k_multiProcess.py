import argparse
import os
import torch
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import cpu_count, set_start_method
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set the start method to 'spawn' to avoid CUDA initialization issues
try:
    set_start_method('spawn')
except RuntimeError:
    # Method already set
    pass
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import MATH_DIFFICULTY_PROMPT

def init_llama_model(model_name="/mnt/cephfs/sumin/model/Llama-3.1-8B-Instruct", device="cuda"):
    """
    Initialize the Llama 3.1 model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        device: Device to load the model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if CUDA is available when device is "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate precision based on device
    if device == "cuda":
        # Use BF16 precision for faster inference on compatible GPUs
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        # Use FP32 precision for CPU
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
    
    return model, tokenizer

def call_llama_llm(model, tokenizer, prompt, system_prompt, n=1, temperature=0.7, max_tokens=32):
    """
    Call Llama 3.1 model for text generation
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The input prompt
        system_prompt: System instruction for the model
        n: Number of generations to perform
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
    
    Returns:
        List of generated texts
    """
    # Format the prompt for Llama 3.1
    formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
    
    results = []
    for _ in range(8):
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate with sampling for diverse outputs
        with torch.no_grad():  # Explicitly disable gradient calculation
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

def difficulty_fn(example, model, tokenizer, math_difficulty_prompt):
    """
    Calculate difficulty for a single example
    
    Args:
        example: Dictionary containing problem data
        model: The loaded model
        tokenizer: The loaded tokenizer
        math_difficulty_prompt: The system prompt for math difficulty assessment
        
    Returns:
        Updated example with difficulty score
    """
    # Skip if already computed
    if 'difficulty' in example and example['difficulty'] is not None:
        return example

    # Extract problem text - only use the question
    problem_text = example.get('question', '')

    # Call LLM for difficulty estimates
    output_list = call_llama_llm(
        model=model,
        tokenizer=tokenizer,
        prompt=f"Problem: {problem_text}",
        system_prompt=math_difficulty_prompt,
        n=8,
        temperature=0.7,
    )
    
    # Filter out any error responses
    output_list = [
        o for o in output_list
        if 'error' not in o.lower()
    ]

    # Attempt to parse each string as float
    values = []
    for o in output_list:
        try:
            # First method: Extract first number from string
            import re
            number_match = re.search(r'(\d+\.?\d*)', o)
            if number_match:
                val = float(number_match.group(1))
                values.append(val)
                continue
            
            # Second method: Remove Llama model's end tokens
            if '|</|assistant>' in o:
                cleaned_str = o.split('|</|assistant>')[0].strip()
                val = float(cleaned_str)
                values.append(val)
                continue
                
            # Third method: Original attempt
            val = float(o)
            values.append(val)
        except ValueError:
            # Ignore anything that can't be parsed as float
            pass

    # Compute the average or set None if no valid floats
    if values:
        difficulty = sum(values) / len(values)
        # Round to nearest integer
        difficulty = round(difficulty)
    else:
        difficulty = None
        print(f'Failed parsing all difficulties: {output_list}')

    example['difficulty'] = difficulty
    return example

def process_chunk(chunk_data, process_idx, math_difficulty_prompt, model_name, device):
    """
    Process a chunk of the dataset
    
    Args:
        chunk_data: List of examples to process
        process_idx: Index of the current process
        math_difficulty_prompt: System prompt for difficulty assessment
        model_name: HuggingFace model name
        device: Device to run the model on
        
    Returns:
        List of processed examples
    """
    print(f"Process {process_idx} starting - handling {len(chunk_data)} examples on {device}")
    
    # Initialize the model within this process on the assigned GPU
    model, tokenizer = init_llama_model(model_name=model_name, device=device)
    
    # Process each example in the chunk
    results = []
    for i, example in enumerate(tqdm(chunk_data, desc=f"Process {process_idx} on {device}")):
        try:
            processed_example = difficulty_fn(example, model, tokenizer, math_difficulty_prompt)
            results.append(processed_example)
            
            # Log progress
            if (i + 1) % 10 == 0:
                print(f"Process {process_idx} on {device}: Completed {i + 1}/{len(chunk_data)} examples")
                
        except Exception as e:
            print(f"Process {process_idx} on {device}: Error processing example {i}: {e}")
            # Still add the original example to maintain dataset structure
            results.append(example)
    
    print(f"Process {process_idx} on {device} completed processing {len(results)} examples")
    return results

def process_gsm8k_dataset(output_file=None, math_difficulty_prompt=None, model_name=None, device="cuda", num_workers=None, debug_mode=False, debug_samples=8):
    """
    Process the gsm8k dataset to compute difficulty scores using ProcessPoolExecutor
    
    Args:
        output_file: Path to save the output dataset
        math_difficulty_prompt: System prompt for difficulty assessment
        model_name: HuggingFace model name for Llama
        device: Device to run the model on (cuda or cpu)
        num_workers: Number of worker processes (default: GPU count if available)
        debug_mode: If True, only process a small number of examples for debugging
        debug_samples: Number of samples to use in debug mode
    """
    # Check available GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() and device == "cuda" else 0
    
    # Determine number of workers based on available GPUs
    if num_workers is None:
        if gpu_count > 0:
            num_workers = 4  # Use exactly one process per GPU
            print(f"Using {num_workers} processes (one per GPU)")
        else:
            num_workers = min(cpu_count(), 4)  # Fallback for CPU
            print(f"No GPUs available. Using {num_workers} CPU processes")
    else:
        if device == "cuda" and num_workers > gpu_count:
            print(f"Warning: Requested {num_workers} workers but only {gpu_count} GPUs available.")
            print(f"Setting workers to {gpu_count} to ensure one process per GPU.")
            num_workers = gpu_count if gpu_count > 0 else num_workers
    
    # Load the dataset
    print("Loading dataset: openai/gsm8k")
    dataset = load_dataset("openai/gsm8k", 'main')
    train_data = dataset['train']
    
    # For debugging, use only the first few examples
    if debug_mode:
        print(f"DEBUG MODE: Using only the first {debug_samples} examples")
        train_data = train_data.select(range(debug_samples))
    
    print(f"Found {len(train_data)} problems in the dataset")
    
    # Convert to list for splitting
    all_examples = list(train_data)
    
    # Split data into exactly num_workers chunks for one chunk per GPU
    chunk_size = len(all_examples) // num_workers
    remainder = len(all_examples) % num_workers
    
    chunks = []
    start_idx = 0
    
    # Create chunks with balanced distribution of examples
    for i in range(num_workers):
        # Add one extra example to the first 'remainder' chunks if dataset size isn't divisible by num_workers
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(all_examples[start_idx:end_idx])
        start_idx = end_idx
    
    print(f"Split dataset into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk)} examples")
    
    # Create device assignments - one GPU per process
    if device == "cuda" and gpu_count > 0:
        devices = [f"cuda:{i}" for i in range(min(num_workers, gpu_count))]
        # If we have more workers than GPUs (shouldn't happen with our setup), cycle through GPUs
        devices = devices + devices * (num_workers // gpu_count) if num_workers > gpu_count else devices
        devices = devices[:num_workers]
    else:
        # Fall back to CPU if no GPUs
        devices = ["cpu"] * num_workers
        
    print(f"Device assignments: {devices}")
    
    # Process chunks in parallel with fixed GPU assignment
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        future_to_idx = {}  # Keep track of which future corresponds to which process index
        
        for i, (chunk, device_assignment) in enumerate(zip(chunks, devices)):
            process_fn = partial(
                process_chunk,
                chunk,
                i,
                math_difficulty_prompt,
                model_name,
                device_assignment  # Pass the specific GPU assignment
            )
            future = executor.submit(process_fn)
            futures.append(future)
            future_to_idx[future] = i
        
        # Prepare a list to collect results in the correct order
        ordered_results = [None] * len(chunks)
        completed_processes = 0
        
        # As futures complete (in any order)
        for future in futures:  # Now we just wait for all to complete
            process_idx = future_to_idx[future]
            try:
                chunk_results = future.result()
                
                # Store results in the correct position based on process index
                ordered_results[process_idx] = chunk_results
                completed_processes += 1
                
                print(f"Process {process_idx} on {devices[process_idx]} completed ({completed_processes}/{len(chunks)} done)")
                
                # Save intermediate results if desired (now they might be partial)
                if completed_processes % 1 == 0 or completed_processes == len(chunks):
                    # Flatten only the completed results so far
                    intermediate_results = []
                    for results in ordered_results:
                        if results is not None:
                            intermediate_results.extend(results)
                    
                    partial_dataset = Dataset.from_list(intermediate_results)
                    partial_path = f"{output_file}_partial_{len(intermediate_results)}"
                    partial_dataset.save_to_disk(partial_path)
                    print(f"Saved partial results with {len(intermediate_results)} examples to {partial_path}")
            except Exception as e:
                print(f"Error in process {process_idx} on {devices[process_idx]}: {e}")
                # Create an empty list for this chunk to maintain order
                ordered_results[process_idx] = []
        
        # Combine all results in the correct order
        all_results = []
        for i, chunk_results in enumerate(ordered_results):
            if chunk_results:  # Skip if the process failed completely
                print(f"Adding {len(chunk_results)} results from process {i}")
                all_results.extend(chunk_results)
    
    # Create the final dataset from all processed results
    final_dataset = Dataset.from_list(all_results)
    
    # Save the final dataset
    if output_file:
        final_path = output_file
    else:
        final_path = "gsm8k_with_difficulty"
    
    final_dataset.save_to_disk(final_path)
    print(f"Finished processing {len(all_results)} entries. Results saved to {final_path}.")
    
    # Verify the saved dataset can be loaded
    print("Verifying the saved dataset can be loaded...")
    loaded_dataset = Dataset.load_from_disk(final_path)
    print(f"Successfully loaded dataset with {len(loaded_dataset)} entries.")
    
    return final_dataset
    
    # Create the final dataset from all processed results
    final_dataset = Dataset.from_list(all_results)
    
    # Save the final dataset
    if output_file:
        final_path = output_file
    else:
        final_path = "gsm8k_with_difficulty"
    
    final_dataset.save_to_disk(final_path)
    print(f"Finished processing {len(all_results)} entries. Results saved to {final_path}.")
    
    # Verify the saved dataset can be loaded
    print("Verifying the saved dataset can be loaded...")
    loaded_dataset = Dataset.load_from_disk(final_path)
    print(f"Successfully loaded dataset with {len(loaded_dataset)} entries.")
    
    return final_dataset

if __name__ == "__main__":
    # This is crucial for CUDA multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Label approximate difficulty for math problems in gsm8k dataset.")
    
    # Output file path
    parser.add_argument(
        "--output_file",
        default="gsm8k_with_difficulty",
        help="Path to save the output dataset with difficulty scores"
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
    
    # Number of worker processes
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(CPU count, 4))"
    )
    
    # Placeholder for the math difficulty prompt
    parser.add_argument(
        "--math_difficulty_prompt",
        default="Your task is to rate the difficulty of a math problem on a scale from 1 to 10. Provide only the numeric rating.",
        help="System prompt for difficulty assessment"
    )
    
    # Add debug_mode and debug_samples arguments
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode (process only a small subset of examples)"
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=8,
        help="Number of samples to process in debug mode"
    )
    
    args = parser.parse_args()
    
    # Process the dataset
    process_gsm8k_dataset(
        output_file=args.output_file,
        math_difficulty_prompt=MATH_DIFFICULTY_PROMPT,
        model_name=args.model_name,
        device=args.device,
        num_workers=args.num_workers,
        debug_mode=args.debug_mode,
        debug_samples=args.debug_samples
    )