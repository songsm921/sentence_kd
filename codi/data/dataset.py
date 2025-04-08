import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

class GSM8kDataset(Dataset):
    """
    Dataset for the GSM8k mathematical reasoning task.
    Uses Hugging Face's datasets library to load and process GSM8k data.
    """
    def __init__(self, dataset_name="gsm8k", split="train", tokenizer=None, max_length=512, 
                 exclude_last_step=True, aug_dataset_path=None):
        """
        Initialize the GSM8k dataset.
        
        Args:
            dataset_name: Name of the dataset in Hugging Face Datasets
            split: 'train', 'validation', or 'test'
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            exclude_last_step: Whether to exclude the final CoT step as mentioned in the paper
            aug_dataset_path: Optional path to augmented dataset (GSM8k-Aug)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.exclude_last_step = exclude_last_step
        
        # Load the dataset
        if aug_dataset_path:
            # Load augmented dataset from local path
            self.dataset = load_dataset("json", data_files={
                "train": f"{aug_dataset_path}/train.jsonl",
                "validation": f"{aug_dataset_path}/validation.jsonl",
                "test": f"{aug_dataset_path}/test.jsonl"
            })[split]
        else:
            # Load standard GSM8k from Hugging Face
            if dataset_name == "gsm8k":
                # GSM8k has 'train' and 'test' splits
                if split == "validation":
                    # Create validation set from train split if needed
                    full_train = load_dataset(dataset_name, "main")["train"]
                    train_val_split = full_train.train_test_split(test_size=0.1, seed=42)
                    self.dataset = train_val_split["test"]  # Use 10% as validation
                elif split == "train":
                    full_train = load_dataset(dataset_name, "main")["train"]
                    train_val_split = full_train.train_test_split(test_size=0.1, seed=42)
                    self.dataset = train_val_split["train"]  # Use 90% as train
                else:  # test
                    self.dataset = load_dataset(dataset_name, "main")["test"]
            else:
                # For other datasets that might have different split names
                self.dataset = load_dataset(dataset_name)[split]
        
        # Prepare answer prompts
        self.answer_prefix = "The answer is:"
        
        # Add special tokens for continuous reasoning
        self.bot_token = "<bot>"  # Beginning of continuous thought
        self.eot_token = "<eot>"  # End of continuous thought
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Returns:
            Dictionary containing:
                - question_input_ids, question_attention_mask: encoded question
                - cot_answer_input_ids, cot_answer_attention_mask: encoded CoT + answer (teacher task)
                - answer_input_ids, answer_attention_mask: encoded answer only (student task)
                - teacher_target_position, student_target_position: positions of the token before the answer
        """
        example = self.dataset[idx]
        
        # Extract question, CoT, and answer
        # The field names might differ based on the dataset format
        if "question" in example:
            question = example["question"]
        elif "input" in example:
            question = example["input"]
        else:
            # Adjust field names based on your dataset
            question = example["problem"]
        
        # For CoT, we need to check if it exists in the dataset
        if "cot" in example:
            cot = example["cot"]
        elif "rationale" in example:
            cot = example["rationale"] 
        elif "steps" in example:
            cot = example["steps"]
        else:
            # If no CoT is provided, we need to handle this case
            # For GSM8k, we can use the answer concatenated with the reasoning
            if "answer" in example and "reasoning" in example:
                cot = example["reasoning"]
            else:
                cot = ""  # Empty CoT if not available
        
        # Get the answer
        if "answer" in example:
            if isinstance(example["answer"], str):
                answer = example["answer"]
            else:
                # Some datasets store answers as integers or other types
                answer = str(example["answer"])
        elif "target" in example:
            answer = example["target"]
        elif "output" in example:
            answer = example["output"]
        else:
            # If the answer is embedded in another field, extract it
            answer = example.get("result", "")
            
        # Exclude the final CoT step if specified (as mentioned in the paper)
        if self.exclude_last_step and cot:
            # Split CoT into steps (depends on dataset format)
            if "<<" in cot:
                # Format like: "<<step1>><<step2>><<step3>>"
                cot_steps = cot.split("<<")
                cot_steps = ["<<" + step for step in cot_steps if step]
                if len(cot_steps) > 1:
                    cot_steps = cot_steps[:-1]  # Remove the last step
                cot = "".join(cot_steps)
            elif "\n" in cot:
                # Alternative approach if steps are separated by newlines
                cot_steps = cot.split("\n")
                if len(cot_steps) > 1:
                    cot_steps = cot_steps[:-1]  # Remove the last step
                cot = "\n".join(cot_steps)
            # Add more format handling as needed
        
        # Prepare inputs for both teacher and student tasks
        question_text = question
        cot_answer_text = f"{question}\n{cot}\n{self.answer_prefix} {answer}"
        answer_text = f"{self.answer_prefix} {answer}"
        
        # Encode the inputs
        question_encoding = self.tokenizer(
            question_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        cot_answer_encoding = self.tokenizer(
            cot_answer_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        answer_encoding = self.tokenizer(
            answer_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get the position of the token before the answer (the colon in "The answer is:")
        # This is for knowledge distillation as described in the paper
        
        # For teacher task
        teacher_tokens = self.tokenizer.convert_ids_to_tokens(cot_answer_encoding["input_ids"][0])
        teacher_target_position = -1
        for i in range(len(teacher_tokens) - 1, 0, -1):
            if teacher_tokens[i] == ":" and i + 1 < len(teacher_tokens):
                teacher_target_position = i
                break
        
        # For student task
        student_tokens = self.tokenizer.convert_ids_to_tokens(answer_encoding["input_ids"][0])
        student_target_position = -1
        for i in range(len(student_tokens) - 1, 0, -1):
            if student_tokens[i] == ":" and i + 1 < len(student_tokens):
                student_target_position = i
                break
        
        # Remove the batch dimension added by the tokenizer
        return {
            "question_input_ids": question_encoding["input_ids"][0],
            "question_attention_mask": question_encoding["attention_mask"][0],
            "cot_answer_input_ids": cot_answer_encoding["input_ids"][0],
            "cot_answer_attention_mask": cot_answer_encoding["attention_mask"][0],
            "answer_input_ids": answer_encoding["input_ids"][0],
            "answer_attention_mask": answer_encoding["attention_mask"][0],
            "teacher_target_position": torch.tensor(teacher_target_position, dtype=torch.long),
            "student_target_position": torch.tensor(student_target_position, dtype=torch.long),
        }

def collate_fn(batch):
    """
    Collate function for DataLoader.
    Batches the examples together.
    """
    result = {
        "question_input_ids": torch.stack([x["question_input_ids"] for x in batch]),
        "question_attention_mask": torch.stack([x["question_attention_mask"] for x in batch]),
        "cot_answer_input_ids": torch.stack([x["cot_answer_input_ids"] for x in batch]),
        "cot_answer_attention_mask": torch.stack([x["cot_answer_attention_mask"] for x in batch]),
        "answer_input_ids": torch.stack([x["answer_input_ids"] for x in batch]),
        "answer_attention_mask": torch.stack([x["answer_attention_mask"] for x in batch]),
        "teacher_target_position": torch.stack([x["teacher_target_position"] for x in batch]),
        "student_target_position": torch.stack([x["student_target_position"] for x in batch]),
    }
    
    return result

def create_augmented_dataset(base_dataset_name="gsm8k", output_path="data/GSM8k-Aug", seed=42):
    """
    Helper function to create an augmented dataset based on GSM8k.
    This would typically be done with a large LM like GPT-4.
    
    This is a placeholder function to illustrate the process. In practice,
    you would need access to a model like GPT-4 to generate the augmented examples.
    """
    from datasets import load_dataset
    import json
    import os
    
    # Load the base dataset
    base_dataset = load_dataset(base_dataset_name, "main")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each split
    for split in base_dataset:
        augmented_examples = []
        
        for example in base_dataset[split]:
            # Here you would typically:
            # 1. Send the question to an LLM like GPT-4
            # 2. Get back a detailed CoT solution
            # 3. Extract the answer and format the data
            
            # Placeholder for augmentation logic
            augmented_example = {
                "question": example["question"],
                "cot": "Placeholder for augmented CoT",  # This would be the LLM-generated CoT
                "answer": example["answer"],
            }
            
            augmented_examples.append(augmented_example)
        
        # Save to JSONL file
        with open(f"{output_path}/{split}.jsonl", "w") as f:
            for example in augmented_examples:
                f.write(json.dumps(example) + "\n")