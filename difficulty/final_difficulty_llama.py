import argparse
import json
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import as_completed

# Global variables for Llama model
llama_model = None
llama_tokenizer = None

from prompt import MATH_DIFFICULTY_PROMPT2
# 수학 문제 난이도 평가를 위한 시스템 프롬프트
MATH_DIFFICULTY_PROMPT = """당신은 수학 문제의 난이도를 평가하는 전문가입니다. 
주어진 수학 문제를 읽고 1부터 10까지의 난이도 점수를 매겨주세요.

난이도 기준:
1: 초등학교 수준의 매우 쉬운 문제
2-3: 중학교 수준의 기본 문제
4-5: 고등학교 수준의 보통 수학 문제
6-7: 고등학교 수준의 어려운 문제, 수학 올림피아드 예선 수준
8-9: 대학 수준 또는 수학 올림피아드 결승 수준의 문제
10: 연구 수준의 극도로 어려운 문제

답변은 숫자만 제공해주세요. 설명이나 추가 텍스트 없이 1부터 10 사이의 숫자 하나만 반환하세요.
"""

def init_llama_model(model_name="meta-llama/Llama-3.1-8B-Instruct", device="cuda"):
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
        
        # Set padding token to EOS token if not defined
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
            print("Set padding token to EOS token")
        
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
        try:
            # Simple tokenization without padding (avoiding padding issues)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            # Generate with sampling for diverse outputs
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=torch.ones_like(inputs.input_ids).to(model.device),
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generation (only the newly generated tokens)
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Add to results
            results.append(response.strip())
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            # Return a default fallback value
            results.append("5")  # Midpoint difficulty value as fallback
    
    return results

def evaluate_difficulty(question):
    """
    Evaluate the difficulty of a math problem using Llama 3.1
    
    Args:
        question: Math problem text
        
    Returns:
        Difficulty score (1-10) as an integer
    """
    # Get difficulty estimates
    output_list = call_llama_llm(
        f"Problem: {question}",
        system_prompt=MATH_DIFFICULTY_PROMPT2,
        n=5,  # 5 estimates for balance between accuracy and speed
        temperature=0.7,
    )
    
    # Filter out invalid responses
    valid_scores = []
    for output in output_list:
        # Try to extract a number from the response
        try:
            # Look for the first number in the response
            for token in output.split():
                token = token.strip(',.;:()[]{}"\'')
                if token.isdigit():
                    score = int(token)
                    if 1 <= score <= 10:  # Ensure it's a valid score
                        valid_scores.append(score)
                        break
        except:
            continue
    
    # Return the average difficulty if there are valid scores, otherwise default to 5
    if valid_scores:
        avg_difficulty = sum(valid_scores) / len(valid_scores)
        return round(avg_difficulty)  # Round to nearest integer
    else:
        print(f"Warning: Could not extract valid scores from responses: {output_list}")
        return 5  # Default to middle difficulty if parsing fails

def update_text_field(row, difficulty):
    """
    Update the text field with the difficulty score in the assistant's response
    
    Args:
        row: DataFrame row with the text field
        difficulty: Evaluated difficulty score
        
    Returns:
        Updated text field
    """
    text = row['text']
    
    # Add difficulty instruction to system prompt
    if "<|im_start|>system" in text:
        system_end_idx = text.find("<|im_end|>", text.find("<|im_start|>system"))
        if system_end_idx != -1:
            modified_system = text[:system_end_idx].strip() + "\nAssess the difficulty level of the problem and state it at the beginning of your answer." + text[system_end_idx:]
            text = modified_system
    
    # Add difficulty score after assistant think
    assistant_think_marker = "<|im_start|>assistant\n<|im_start|>think"
    if assistant_think_marker in text:
        idx = text.find(assistant_think_marker) + len(assistant_think_marker)
        difficulty_text = f"\nDifficulty: {difficulty}/10\n\n"
        text = text[:idx] + difficulty_text + text[idx:]
    
    return text

def process_s1k_dataset(input_path, output_path, model_name="meta-llama/Llama-3.1-8B-Instruct", 
                       device="cuda", num_workers=4, hf_dataset=True):
    """
    Process the s1K dataset to evaluate difficulty of math problems
    
    Args:
        input_path: Path to the input dataset (HF dataset name or local path)
        output_path: Path to save the modified dataset
        model_name: Name of the LLM model to use
        device: Device to run on (cuda or cpu)
        num_workers: Number of concurrent workers
        hf_dataset: Whether input is a Hugging Face dataset name or local file
    """
    # Initialize the model
    init_llama_model(model_name, device)
    
    # Load the dataset
    print(f"Loading dataset from {input_path}")
    if hf_dataset:
        # Load from Hugging Face Hub
        try:
            dataset = load_dataset(input_path)
            # Convert to DataFrame for easier processing
            if isinstance(dataset, DatasetDict):
                # If the dataset has splits, use the 'train' split or the first available split
                if 'train' in dataset:
                    df = dataset['train'].to_pandas()
                else:
                    first_split = list(dataset.keys())[0]
                    df = dataset[first_split].to_pandas()
                    print(f"Using '{first_split}' split from dataset")
            else:
                df = dataset.to_pandas()
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            raise
    else:
        # Load from local file
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Filter only math problems
    math_df = df[df['cot_type'] == 'math'].copy()
    print(f"Math problems: {len(math_df)} rows")
    
    # Add difficulty column to store the evaluated difficulty
    math_df['difficulty'] = None
    
    # Function to process each math problem
    def process_row(idx):
        row = math_df.iloc[idx]
        question = row['question']
        
        # Evaluate difficulty
        difficulty = evaluate_difficulty(question)
        
        # Update text field with difficulty information
        updated_text = update_text_field(row, difficulty)
        
        return idx, difficulty, updated_text
    
    # Process in parallel
    print("Evaluating difficulty and updating dataset...")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_row, i): i for i in range(len(math_df))}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, difficulty, updated_text = future.result()
            results[idx] = (difficulty, updated_text)
    
    # Update the DataFrame
    for idx, (difficulty, updated_text) in results.items():
        math_df.at[math_df.index[idx], 'text'] = updated_text
        math_df.at[math_df.index[idx], 'difficulty'] = difficulty
    
    # Convert DataFrame back to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(math_df)
    
    # Save as Hugging Face dataset for compatibility with load_dataset
    print(f"Saving updated dataset with {len(math_df)} rows to {output_path}")
    hf_dataset.save_to_disk(output_path)
    
    # Also save a copy as Parquet file for convenience
    parquet_path = os.path.join(output_path, "data.parquet")
    os.makedirs(output_path, exist_ok=True)
    math_df.to_parquet(parquet_path, index=False)
    print(f"Saved additional copy as Parquet file: {parquet_path}")
    
    print("Processing completed successfully!")
    
    # Return some statistics
    difficulties = [d for _, (d, _) in results.items()]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
    difficulty_counts = {i: difficulties.count(i) for i in range(1, 11)}
    
    print(f"Average difficulty: {avg_difficulty:.2f}")
    print("Difficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  Level {diff}: {count} problems ({count/len(difficulties)*100:.1f}%)")
    
    return hf_dataset

def load_difficulty_dataset(dataset_path):
    """
    Helper function to load the processed dataset with difficulty scores
    
    Args:
        dataset_path: Path to the saved dataset directory
        
    Returns:
        Loaded Hugging Face dataset
    """
    print(f"Loading difficulty-evaluated dataset from {dataset_path}")
    return load_dataset(dataset_path, data_files=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate difficulty of math problems in s1K dataset")
    parser.add_argument("--input", required=True, 
                        help="Input dataset: either a Hugging Face dataset name (simplescaling/s1K_tokenized) or local file path")
    parser.add_argument("--output", required=True, 
                        help="Directory path to save the modified dataset for use with load_dataset")
    parser.add_argument("--model", default="/mnt/cephfs/sumin/model/Llama-3.1-8B-Instruct", 
                        help="HuggingFace model name or path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Device to run the model on")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of worker threads for parallel processing")
    parser.add_argument("--local", action="store_true",
                        help="Treat input as local file instead of Hugging Face dataset name")
    
    args = parser.parse_args()
    
    dataset = process_s1k_dataset(
        args.input, 
        args.output, 
        model_name=args.model, 
        device=args.device,
        num_workers=args.workers,
        hf_dataset=not args.local
    )
    
    print("\nExample of how to load the processed dataset:")
    print(f"from datasets import load_dataset")
    print(f"dataset = load_dataset('{args.output}')")
    print(f"# OR")
    print(f"dataset = load_dataset('arrow', data_files='{args.output}/data.arrow')")