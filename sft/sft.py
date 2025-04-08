import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, load_from_disk
import transformers
import trl
import random

@dataclass
class TrainingConfig:
    model_name: str = field(default="/mnt/cephfs/echoi/models/Qwen2.5-7B-Instruct/")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    local_train_file: bool = field(default=False)
    # 난이도 관련 설정 추가
    add_difficulty_instruction: bool = field(default=True)
    example_count: int = field(default=100)  # 예시 난이도를 추가할 데이터 수
    
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def add_system_instruction(dataset, config):
    """시스템 프롬프트에 난이도 평가 지시를 추가합니다."""
    
    def modify_system(example):
        text = example["text"]
        
        # 시스템 프롬프트가 있는지 확인
        if "<|im_start|>system" in text and "<|im_end|>" in text:
            system_start = text.find("<|im_start|>system")
            system_end = text.find("<|im_end|>", system_start)
            
            if system_end != -1:
                system_content = text[system_start:system_end]
                
                # 난이도 평가 지시 추가 (영어)
                difficulty_instruction = "\nAdditionally, after solving each problem, assess its difficulty on a scale of 1-10, where 1 is the easiest and 10 is the most difficult, and include 'Difficulty: X' at the end of your answer."
                
                # 난이도 지시가 이미 있는지 확인 (중복 방지)
                if "difficulty" not in system_content.lower() and "Difficulty" not in system_content:
                    modified_system = system_content + difficulty_instruction
                    modified_text = text[:system_start] + modified_system + text[system_end:]
                    return {"text": modified_text}
                
        # 시스템 프롬프트가 없는 경우 생성
        elif "<|im_start|>user" in text:
            user_start = text.find("<|im_start|>user")
            
            # 시스템 프롬프트 추가 (영어)
            system_prompt = "<|im_start|>system\nYou are a helpful problem-solving assistant. Solve the given problems step by step and provide clear explanations.\nAdditionally, after solving each problem, assess its difficulty on a scale of 1-10, where 1 is the easiest and 10 is the most difficult, and include 'Difficulty: X' at the end of your answer.<|im_end|>\n"
            modified_text = system_prompt + text
            return {"text": modified_text}
        
        return example
    
    if config.add_difficulty_instruction:
        return dataset.map(modify_system)
    return dataset

def add_user_instruction(dataset, config):
    """User 프롬프트에 난이도 평가 지시를 추가합니다."""
    
    def modify_prompt(example):
        text = example["text"]
        
        # User 프롬프트 확인
        if "<|im_start|>user" in text and "<|im_end|>" in text:
            user_start = text.find("<|im_start|>user")
            user_end = text.find("<|im_end|>", user_start)
            
            if user_end != -1:
                user_content = text[user_start:user_end]
                
                # 난이도 평가 지시 추가 (영어)
                difficulty_instruction = "\n\nAfter solving this problem, please rate its difficulty on a scale of 1-10 and include 'Difficulty: X' at the end of your answer."
                
                # 난이도 지시가 이미 있는지 확인 (중복 방지)
                if "difficulty" not in user_content.lower() and "Difficulty" not in user_content:
                    modified_user = user_content + difficulty_instruction
                    modified_text = text[:user_start] + modified_user + text[user_end:]
                    return {"text": modified_text}
        
        return example
    
    if config.add_difficulty_instruction:
        return dataset.map(modify_prompt)
    return dataset

def add_example_difficulties(dataset, config):
    """일부 데이터에 예시 난이도 평가를 추가합니다."""
    
    if not config.add_difficulty_instruction or config.example_count <= 0:
        return dataset
    
    modified_examples = []
    example_count = min(config.example_count, len(dataset["train"]))
    
    for i, example in enumerate(dataset["train"]):
        if i >= example_count:
            break
            
        text = example["text"]
        if "<|im_start|>assistant" in text and "<|im_end|>" in text:
            assistant_start = text.find("<|im_start|>assistant")
            assistant_end = text.find("<|im_end|>", assistant_start)
            
            if assistant_end != -1:
                assistant_content = text[assistant_start:assistant_end]
                
                # 문제 난이도 결정 (여기서는 예시로 랜덤 값 사용)
                # 실제로는 문제 복잡성을 분석하는 로직을 사용할 수 있습니다
                difficulty = random.randint(1, 10)
                
                # 난이도 평가가 이미 있는지 확인 (중복 방지)
                if "Difficulty:" not in assistant_content:
                    # 최종 답변 다음에 난이도 추가
                    if assistant_content.strip().endswith("}"):
                        # 수학 문제의 경우 마지막 수식 다음에 추가
                        modified_assistant = assistant_content.rstrip() + f"\n\nDifficulty: {difficulty}"
                    else:
                        # 일반 텍스트의 경우
                        lines = assistant_content.split('\n')
                        # 마지막 줄이 비어있지 않은 경우
                        if lines and lines[-1].strip():
                            modified_assistant = assistant_content.rstrip() + f"\n\nDifficulty: {difficulty}"
                        else:
                            # 마지막 줄이 비어있는 경우
                            modified_assistant = assistant_content.rstrip() + f"Difficulty: {difficulty}"
                    
                    modified_text = text[:assistant_start] + modified_assistant + text[assistant_end:]
                    # print(modified_text)
                    modified_examples.append({"text": modified_text})
                    continue
        
        # 수정하지 않은 예제는 그대로 추가
        modified_examples.append(example)
    
    # 나머지 데이터셋 유지
    remaining_examples = [example for i, example in enumerate(dataset["train"]) if i >= example_count]
    
    # 수정된 예시와 나머지 데이터셋 결합
    modified_train_dataset = Dataset.from_list(modified_examples + remaining_examples)
    
    # 테스트 데이터셋 처리
    test_dataset = dataset["test"] if "test" in dataset else dataset["train"]
    
    return DatasetDict({"train": modified_train_dataset, "test": test_dataset})

def preprocess_dataset(dataset, config):
    """데이터셋 전처리 단계를 순차적으로 적용합니다."""
    
    # 1. 시스템 프롬프트에 난이도 평가 지시 추가
    dataset = add_system_instruction(dataset, config)
    
    # 2. 사용자 프롬프트에 난이도 평가 지시 추가
    # dataset = add_user_instruction(dataset, config)
    
    # 3. 일부 데이터에 예시 난이도 평가 추가
    dataset = add_example_difficulties(dataset, config)
    
    return dataset

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # 데이터셋 로드
    if config.local_train_file:
        logging.info('Loading local train file...')
        dataset = load_from_disk("/mnt/cephfs/sumin/sentence_kd/dataset/stable/mk4_before_think")
    else:
        dataset = load_dataset(config.train_file_path)
    
    # 데이터셋 전처리 (난이도 평가 지시 및 예시 추가)
    # processed_dataset = preprocess_dataset(dataset, config)
    
    # 데이터셋 샘플 로깅 (디버깅용)
    # logging.info(f"Processed dataset sample: {processed_dataset['train'][0]}")
    
    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    
    # Only compute loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )
    
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()