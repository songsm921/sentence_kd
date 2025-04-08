import argparse
import concurrent.futures
import os
import torch
import pandas as pd
from concurrent.futures import as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import MATH_DIFFICULTY_PROMPT
# from deepscaler.system_prompts import MATH_DIFFICULTY_PROMPT
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
        # llama_tokenizer.pad_token = "[PAD]"
        # llama_tokenizer.padding_side = "left"
        
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

def call_llama_llm(prompt, system_prompt, n=1, temperature=0.7, max_tokens=32):
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
    # print(formatted_prompt)
    results = []
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
        print(response)
        # Add to results
        results.append(response.strip())
    
    return results

def difficulty_fn(idx, entry, math_difficulty_prompt):
    """
    1) Extract problem text (only the question, not the solution).
    2) Call LLM for difficulty estimates (8 numeric strings).
    3) Convert to float safely, filter out parse errors.
    4) Take the average and store as 'difficulty'.
    
    Args:
        idx: Index of the entry
        entry: Dictionary containing problem data
        math_difficulty_prompt: The system prompt for math difficulty assessment
    """
    # Skip if already computed
    if 'difficulty' in entry and entry['difficulty'] is not None:
        return idx, entry

    # Extract problem text - only use the question
    problem_text = entry.get('question', '')

    # Placeholder for the prompt - this will be filled in by the user
    # Pass@8 difficulty calls using Llama
    # print(problem_text)
    output_list = call_llama_llm(
        f"Problem: {problem_text}",
        system_prompt=math_difficulty_prompt,
        n=8,
        temperature=0.7,
    )
    print(output_list)
    
    # Filter out any error responses
    output_list = [
        o for o in output_list
        if 'error' not in o.lower()
    ]

    # Attempt to parse each string as float
    values = []
    for o in output_list:
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
        print(f'Failed parsing all difficulties for index {idx}: {output_list}')

    entry['difficulty'] = difficulty
    print(f"Index {idx}: Difficulty = {difficulty}")
    return idx, entry

def process_s1k_dataset(output_file=None, math_difficulty_prompt=None, model_name=None, device="cuda"):
    """
    Process the simplescaling/s1K_tokenized dataset to compute difficulty scores
    for math problems and save the filtered dataset
    
    Args:
        output_file: Path to save the output dataset
        math_difficulty_prompt: System prompt for difficulty assessment
        model_name: HuggingFace model name for Llama
        device: Device to run the model on (cuda or cpu)
    """
    # Initialize the Llama model
    if model_name:
        init_llama_model(model_name=model_name, device=device)
    else:
        init_llama_model(device=device)
    
    # Load the dataset
    print("Loading dataset: simplescaling/s1K_tokenized")
    dataset = load_dataset("simplescaling/s1K_tokenized")
    
    # Extract the training split
    train_data = dataset['train']
    
    # Filter for math problems
    print("Filtering for math problems...")
    math_data = [example for example in train_data if example['cot_type'] == 'math']
    
    print(f"Found {len(math_data)} math problems in the dataset")
    
    # Use ThreadPoolExecutor to process concurrently
    print("Starting difficulty assessment...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(difficulty_fn, i, example, math_difficulty_prompt)
            for i, example in enumerate(math_data)
        ]
        
        # Track progress and save intermediate results
        results = [None] * len(math_data)
        done_count = 0
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            done_count += 1

            # Periodically save partial results (every 500 entries)
            if done_count % 500 == 0:
                print(f"Processed {done_count} entries so far. Saving partial results...")
                valid_results = [r for r in results if r is not None]
                partial_dataset = Dataset.from_list(valid_results)
                partial_path = f"{output_file}_partial_{done_count}"
                partial_dataset.save_to_disk(partial_path)
                print(f"Partial results saved to {partial_path}")
    
    # Create the final dataset from the processed results
    final_dataset = Dataset.from_list(results)
    
    # Save the final dataset
    if output_file:
        final_path = output_file
    else:
        final_path = "s1k_math_with_difficulty"
    
    final_dataset.save_to_disk(final_path)
    print(f"Finished processing {len(results)} entries. Results saved to {final_path}.")
    
    # Verify the saved dataset can be loaded
    print("Verifying the saved dataset can be loaded...")
    loaded_dataset = Dataset.load_from_disk(final_path)
    print(f"Successfully loaded dataset with {len(loaded_dataset)} entries.")
    
    return final_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label approximate difficulty for math problems in s1K dataset.")
    
    # Output file path
    parser.add_argument(
        "--output_file",
        default="s1k_math_with_difficulty",
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
        device=args.device
    )