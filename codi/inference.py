import argparse
import torch
from transformers import AutoTokenizer
from models.codi_model import CODI
from utils.visualization import visualize_continuous_thoughts, visualize_attention

def parse_args():
    parser = argparse.ArgumentParser(description="CODI Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--num_latent", type=int, default=6, help="Number of continuous thought tokens")
    parser.add_argument("--question", type=str, required=True, help="Question to process")
    parser.add_argument("--visualize", action="store_true", help="Visualize continuous thoughts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model
    model = CODI(
        model_name=args.model_name,
        num_latent=args.num_latent,
        bot_token_id=tokenizer.convert_tokens_to_ids("<bot>"),
        eot_token_id=tokenizer.convert_tokens_to_ids("<eot>"),
        tokenizer=tokenizer,
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        args.question,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(args.device)
    
    # Generate answer
    print(f"Question: {args.question}")
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            inputs.attention_mask
        )
        
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated answer: {answer}")
        
        # Decode continuous thoughts for interpretability
        from utils.trainer import decode_continuous_thoughts
        thought_results = decode_continuous_thoughts(model, args.question, tokenizer, args.device, top_k=5)
        
        print("\nDecoded Continuous Thoughts:")
        for i, result in enumerate(thought_results):
            tokens = result['tokens']
            values = result['values']
            print(f"Thought {i+1}: {' | '.join([f'{t} ({v:.2f})' for t, v in zip(tokens, values)])}")
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_continuous_thoughts(model, args.question, tokenizer, args.device, "continuous_thoughts.png")
        try:
            visualize_attention(model, args.question, tokenizer, args.device, "attention.png")
            print("Visualizations saved as continuous_thoughts.png and attention.png")
        except:
            print("Could not generate attention visualization (requires model modification)")
            print("Visualization saved as continuous_thoughts.png")

if __name__ == "__main__":
    main()