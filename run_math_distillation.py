#!/usr/bin/env python3
"""
Script to run the complete math dataset creation and knowledge distillation pipeline.
This connects the dataset composition with the knowledge distillation training.
"""

import os
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the math dataset and distillation pipeline")
    
    # Dataset configuration
    parser.add_argument("--datasets", type=str, nargs="+", 
                        choices=["gsm8k", "open_platypus", "aime", "math"], 
                        default=["gsm8k", "math"],
                        help="List of datasets to include")
    
    parser.add_argument("--math_level", type=str, default="all",
                        choices=["all", "easy", "medium", "hard", "very_hard"],
                        help="Difficulty level for MATH dataset")
    
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples per dataset (for testing)")
    
    # Distillation configuration
    parser.add_argument("--teacher_model_name", type=str, default="meta-llama/Llama-2-13b-hf", 
                        help="Teacher model name or path")
    
    parser.add_argument("--student_model_name", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Student model name or path")
    
    parser.add_argument("--distillation_type", type=str, 
                        choices=["token", "sentence", "hybrid"], 
                        default="hybrid", 
                        help="Type of distillation to use")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    
    parser.add_argument("--output_dir", type=str, default="./math_distilled_model",
                        help="Output directory for model checkpoints")
    
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    parser.add_argument("--evaluation_dataset", type=str, default=None,
                        choices=["iwslt14", "iwslt13", "wmt14", "iwslt17", None],
                        help="Dataset to evaluate on (optional)")
    
    return parser.parse_args()

def run_dataset_composition(args):
    """Run the dataset composition step"""
    logger.info("Step 1: Creating math problem dataset...")
    
    dataset_cmd = [
        "python", "dataset_composition.py",
        "--datasets", *args.datasets,
        "--math_level", args.math_level,
        "--max_seq_length", str(args.max_seq_length),
        "--train_file", os.path.join(args.output_dir, "train_data.txt"),
        "--output_file", os.path.join(args.output_dir, "math_dataset.json"),
        "--tokenizer_name", args.student_model_name,
    ]
    
    if args.max_samples is not None:
        dataset_cmd.extend(["--max_samples", str(args.max_samples)])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Running command: {' '.join(dataset_cmd)}")
    subprocess.run(dataset_cmd)
    
    train_file = os.path.join(args.output_dir, "train_data.txt")
    if not os.path.exists(train_file):
        logger.error(f"Training file {train_file} was not created! Aborting.")
        return False
    
    logger.info(f"Dataset created. Training file: {train_file}")
    return True

def run_knowledge_distillation(args):
    """Run the knowledge distillation training step"""
    logger.info("Step 2: Running knowledge distillation training...")
    
    train_file = os.path.join(args.output_dir, "train_data.txt")
    
    distill_cmd = [
        "python", "knowledge_distillation.py",
        "--teacher_model_name", args.teacher_model_name,
        "--student_model_name", args.student_model_name,
        "--train_file", train_file,
        "--output_dir", args.output_dir,
        "--max_seq_length", str(args.max_seq_length),
        "--batch_size", str(args.batch_size),
        "--distillation_type", args.distillation_type,
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_train_epochs),
    ]
    
    if args.fp16:
        distill_cmd.append("--fp16")
    
    logger.info(f"Running command: {' '.join(distill_cmd)}")
    subprocess.run(distill_cmd)
    
    # Check if the final model was created
    final_model_dir = os.path.join(args.output_dir, "final")
    if not os.path.exists(final_model_dir):
        logger.error(f"Final model directory {final_model_dir} was not created! Distillation may have failed.")
        return False
    
    logger.info(f"Knowledge distillation completed. Model saved to {final_model_dir}")
    return True

def run_evaluation(args):
    """Optionally run evaluation if requested"""
    if args.evaluation_dataset is None:
        logger.info("No evaluation dataset specified, skipping evaluation.")
        return True
    
    logger.info(f"Step 3: Evaluating model on {args.evaluation_dataset}...")
    
    final_model_dir = os.path.join(args.output_dir, "final")
    eval_cmd = [
        "python", "evaluate.py",
        "--teacher_model_path", args.teacher_model_name,
        "--student_model_path", final_model_dir,
        "--dataset", args.evaluation_dataset,
        "--src_lang", "en",  # Default to English source
        "--tgt_lang", "de",  # Default to German target
        "--output_file", os.path.join(args.output_dir, "evaluation_results.json"),
    ]
    
    if args.fp16:
        eval_cmd.append("--fp16")
    
    logger.info(f"Running command: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd)
    
    # Check if the evaluation results file was created
    eval_file = os.path.join(args.output_dir, "evaluation_results.json")
    if not os.path.exists(eval_file):
        logger.error(f"Evaluation results file {eval_file} was not created! Evaluation may have failed.")
        return False
    
    logger.info(f"Evaluation completed. Results saved to {eval_file}")
    return True

def main():
    args = parse_args()
    
    logger.info("Starting math knowledge distillation pipeline")
    
    # Step 1: Create the dataset
    if not run_dataset_composition(args):
        return
    
    # Step 2: Run knowledge distillation
    if not run_knowledge_distillation(args):
        return
    
    # Step 3: Optionally run evaluation
    run_evaluation(args)
    
    logger.info("Math knowledge distillation pipeline completed successfully!")

if __name__ == "__main__":
    main()