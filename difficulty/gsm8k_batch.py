import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import MATH_DIFFICULTY_PROMPT

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

def call_llama_llm(prompts, system_prompt, temperature=0.7, max_tokens=32):
    """
    Call Llama 3.1 model for text generation in batch
    
    Args:
        prompts: List of input prompts
        system_prompt: System instruction for the model
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
    
    Returns:
        List of generated texts for each prompt (8 outputs per prompt)
    """
    model, tokenizer = init_llama_model()
    
    all_results = []
    
    for prompt in prompts:
        # Format the prompt for Llama 3.1
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
        
        prompt_results = []
        for _ in range(8):
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
            # print(response)
            # Add to results
            prompt_results.append(response.strip())
        
        all_results.append(prompt_results)
    
    return all_results

def process_batch(batch_indices, batch_entries, math_difficulty_prompt):
    """
    Process a batch of examples at once
    
    Args:
        batch_indices: List of indices for the batch
        batch_entries: List of entries to process
        math_difficulty_prompt: System prompt for difficulty assessment
    
    Returns:
        List of tuples (idx, processed_entry)
    """
    # Extract problem texts from the batch
    problem_texts = []
    for entry in batch_entries:
        problem_text = entry.get('question', '')
        problem_texts.append(f"Problem: {problem_text}")
    
    # Call LLM for all prompts in the batch
    all_outputs = call_llama_llm(
        prompts=problem_texts,
        system_prompt=math_difficulty_prompt,
        temperature=0.7,
    )
    
    # Process the outputs for each example
    results = []
    for idx, entry, outputs in zip(batch_indices, batch_entries, all_outputs):
        # Skip if already computed
        if 'difficulty' in entry and entry['difficulty'] is not None:
            results.append((idx, entry))
            continue
        
        # Make a copy of the entry to avoid modifying the original
        entry_copy = dict(entry)
        
        print(outputs)
        # Filter out any error responses
        filtered_outputs = [
            o for o in outputs
            if 'error' not in o.lower()
        ]
        
        # Attempt to parse each string as float
        values = []
        for o in filtered_outputs:
            try:
                # 첫 번째 방법: 문자열에서 첫 번째 숫자만 추출
                import re
                number_match = re.search(r'(\d+\.?\d*)', o)
                if number_match:
                    val = float(number_match.group(1))
                    values.append(val)
                    continue
                
                # 두 번째 방법: Llama 모델의 종료 토큰 제거
                if '|</|assistant>' in o:
                    cleaned_str = o.split('|</|assistant>')[0].strip()
                    val = float(cleaned_str)
                    values.append(val)
                    continue
                    
                # 세 번째 방법: 원래 시도
                val = float(o)
                values.append(val)
            except ValueError:
                # Ignore anything that can't be parsed as float
                pass
        
        # Compute the average or set None if no valid floats
        if values:
            difficulty = sum(values) / len(values)
            # Round to 1 decimal place
            difficulty = round(difficulty)
        else:
            difficulty = None
            print(f'Failed parsing all difficulties for index {idx}: {filtered_outputs}')
        
        entry_copy['difficulty'] = difficulty
        print(f"Index {idx}: Difficulty = {difficulty}")
        
        results.append((idx, entry_copy))
    
    return results

def process_s1k_dataset(output_file=None, math_difficulty_prompt=None, model_name=None, device="cuda", batch_size=4):
    """
    Process the GSM8K dataset to compute difficulty scores
    for math problems and save the filtered dataset using batch processing
    
    Args:
        output_file: Path to save the output dataset
        math_difficulty_prompt: System prompt for difficulty assessment
        model_name: HuggingFace model name for Llama
        device: Device to run the model on (cuda or cpu)
        batch_size: Number of examples to process in each batch
    """
    # Initialize the Llama model
    if model_name:
        init_llama_model(model_name=model_name, device=device)
    else:
        init_llama_model(device=device)
    
    # Load the dataset
    print("Loading dataset: openai/gsm8k")
    dataset = load_dataset("openai/gsm8k", 'main')
    
    # Extract the training split
    train_data = dataset['train']
    
    print(f"Found {len(train_data)} math problems in the dataset")
    
    # Create batches
    num_examples = len(train_data)
    batch_indices = list(range(0, num_examples, batch_size))
    
    # Initialize results array
    results = [None] * num_examples
    
    # Process batches
    print("Starting batch processing...")
    for i in tqdm(range(0, len(batch_indices))):
        start_idx = batch_indices[i]
        end_idx = min(start_idx + batch_size, num_examples)
        
        current_indices = list(range(start_idx, end_idx))
        current_entries = [train_data[idx] for idx in current_indices]
        
        # Process the current batch
        batch_results = process_batch(current_indices, current_entries, math_difficulty_prompt)
        
        # Store results
        for idx, result in batch_results:
            results[idx] = result
        
        # Periodically save partial results (every 500 entries or at the end)
        if (end_idx % 500 < batch_size) or (end_idx == num_examples):
            print(f"Processed {end_idx} entries so far. Saving partial results...")
            valid_results = [r[1] for r in results[:end_idx] if r is not None]
            partial_dataset = Dataset.from_list(valid_results)
            partial_path = f"{output_file}_partial_{end_idx}"
            partial_dataset.save_to_disk(partial_path)
            print(f"Partial results saved to {partial_path}")
            
            # Clear CUDA cache to free up memory
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Create the final dataset from the processed results
    final_dataset = Dataset.from_list([r[1] for r in results if r is not None])
    
    # Save the final dataset
    if output_file:
        final_path = output_file
    else:
        final_path = "gsm8k_with_diffculty"
    
    final_dataset.save_to_disk(final_path)
    print(f"Finished processing {len(final_dataset)} entries. Results saved to {final_path}.")
    
    # Verify the saved dataset can be loaded
    print("Verifying the saved dataset can be loaded...")
    loaded_dataset = Dataset.load_from_disk(final_path)
    print(f"Successfully loaded dataset with {len(loaded_dataset)} entries.")
    
    return final_dataset

if __name__ == "__main__":
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
    
    # Batch size for processing
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of examples to process in each batch"
    )
    
    # Placeholder for the math difficulty prompt (to be provided by the user)
    parser.add_argument(
        "--math_difficulty_prompt",
        default="Your task is to rate the difficulty of a math problem on a scale from 1 to 10. Provide only the numeric rating.",
        help="System prompt for difficulty assessment"
    )
    
    args = parser.parse_args()
    
    # Process the dataset
    process_s1k_dataset(
        output_file=args.output_file,
        math_difficulty_prompt=MATH_DIFFICULTY_PROMPT,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )