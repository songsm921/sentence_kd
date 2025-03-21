#!/usr/bin/env python3
"""
Script to run the distillation training with different configurations.
This helps automate the process of running experiments with token-level,
sentence-level, and hybrid distillation.
"""

import os
import subprocess
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run knowledge distillation training with different methods")
    
    # Model configuration
    parser.add_argument("--teacher_model_name", type=str, default="meta-llama/Llama-2-8b-hf", 
                        help="Teacher model name or path")
    parser.add_argument("--student_model_name", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Student model name or path")
    
    # Training configuration
    parser.add_argument("--train_file", type=str, required=True, 
                        help="Path to training data file")
    parser.add_argument("--base_output_dir", type=str, default="./distilled_models", 
                        help="Base output directory for all experiments")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Peak learning rate")
    
    # Experiment configuration
    parser.add_argument("--methods", type=str, nargs="+", 
                        choices=["token", "sentence", "hybrid"], 
                        default=["token", "sentence", "hybrid"],
                        help="Distillation methods to run")
    parser.add_argument("--initial_gate_values", type=float, nargs="+", default=[0.5],
                        help="Initial gate values to try for hybrid distillation")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="Temperatures to try for token-level distillation")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use for training")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create base output directory
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiments for each distillation method
    for method in args.methods:
        logger.info(f"Running experiments for {method} distillation")
        
        if method == "hybrid":
            # For hybrid method, try different initial gate values
            for gate_value in args.initial_gate_values:
                for temp in args.temperatures:
                    run_experiment(args, method, gate_value, temp, timestamp)
        else:
            # For token and sentence methods, only temperature matters for token-level
            for temp in args.temperatures:
                run_experiment(args, method, None, temp, timestamp)

def run_experiment(args, method, gate_value, temperature, timestamp):
    """Run a single distillation experiment with the specified configuration"""
    
    # Create a unique output directory for this experiment
    if method == "hybrid" and gate_value is not None:
        output_dir = os.path.join(
            args.base_output_dir, 
            f"{timestamp}_{method}_gate{gate_value}_temp{temperature}"
        )
    else:
        output_dir = os.path.join(
            args.base_output_dir, 
            f"{timestamp}_{method}_temp{temperature}"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command to run the training script
    cmd = [
        "python", "knowledge_distillation.py",
        "--teacher_model_name", args.teacher_model_name,
        "--student_model_name", args.student_model_name,
        "--train_file", args.train_file,
        "--output_dir", output_dir,
        "--max_seq_length", str(args.max_seq_length),
        "--batch_size", str(args.batch_size),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--distillation_type", method,
        "--temperature", str(temperature),
        "--device", args.device
    ]
    
    # Add gate value for hybrid distillation
    if method == "hybrid" and gate_value is not None:
        cmd.extend(["--initial_gate_value", str(gate_value)])
    
    # Add FP16 flag if requested
    if args.fp16:
        cmd.append("--fp16")
    
    # Convert all arguments to strings
    cmd = [str(c) for c in cmd]
    
    # Log the command
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Capture and log output in real-time
    for line in process.stdout:
        logger.info(line.strip())
    
    # Wait for the process to complete
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"Experiment failed with return code {process.returncode}")
    else:
        logger.info(f"Experiment completed successfully")

if __name__ == "__main__":
    main()