import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from models.codi_model import CODI
from data.dataset import GSM8kDataset, collate_fn
from utils.trainer import train, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="CODI: Continuous Chain-of-Thought via Self-Distillation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model to use for both teacher and student")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name in Hugging Face datasets")
    parser.add_argument("--aug_dataset_path", type=str, default=None, 
                        help="Path to the augmented dataset (if using local files)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save models")
    parser.add_argument("--num_latent", type=int, default=6, help="Number of continuous thought tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--effective_batch_size", type=int, default=128, help="Effective batch size with gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for teacher CE loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for student CE loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for knowledge distillation loss")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and setup special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for continuous reasoning
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens to the tokenizer")
    
    # Create datasets and dataloaders using Hugging Face's datasets library
    print("Loading datasets...")
    train_dataset = GSM8kDataset(
        dataset_name=args.dataset,
        split="train",
        tokenizer=tokenizer,
        exclude_last_step=True,
        aug_dataset_path=args.aug_dataset_path
    )
    
    val_dataset = GSM8kDataset(
        dataset_name=args.dataset,
        split="validation",
        tokenizer=tokenizer,
        exclude_last_step=True,
        aug_dataset_path=args.aug_dataset_path
    )
    
    test_dataset = GSM8kDataset(
        dataset_name=args.dataset,
        split="test",
        tokenizer=tokenizer,
        exclude_last_step=False,
        aug_dataset_path=args.aug_dataset_path
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn
    )
    
    # Calculate gradient accumulation steps
    grad_accum_steps = max(1, args.effective_batch_size // args.batch_size)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")
    
    # Initialize model
    print(f"Initializing CODI model with {args.model} as base model")
    model = CODI(
        model_name=args.model,
        num_latent=args.num_latent,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids("<eot>"),
        tokenizer=tokenizer,
    )
    model.to(args.device)
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Train the model
    print(f"Starting training on {args.device}")
    best_val_accuracy = 0.0
    
    for epoch in range(args.num_epochs):
        train_loss = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            grad_accum_steps=grad_accum_steps,
            epoch=epoch
        )
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_dataloader, args.device)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        
        # Also save a checkpoint for each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Load the best model and evaluate on test set
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    test_loss, test_accuracy = evaluate(model, test_dataloader, args.device, is_test=True)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("Training completed!")

if __name__ == "__main__":
    main()