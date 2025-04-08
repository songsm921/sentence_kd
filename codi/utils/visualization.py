import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

def visualize_continuous_thoughts(model, question, tokenizer, device, save_path=None):
    """
    Visualize the continuous thoughts for a given question as described in the CODI paper.
    Creates a figure showing the top decoded tokens for each continuous thought.
    
    Args:
        model: The CODI model
        question: Question text
        tokenizer: Tokenizer
        device: Device for inference
        save_path: Path to save the visualization (if None, just shows the plot)
    """
    # Get decoded continuous thoughts
    from utils.trainer import decode_continuous_thoughts
    results = decode_continuous_thoughts(model, question, tokenizer, device, top_k=5)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Continuous Thoughts Visualization\nQuestion: {question}", fontsize=14)
    
    # Create grid for visualization
    num_thoughts = len(results)
    
    # Create the main grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, num_thoughts + 1)
    ax.set_yticks(range(1, num_thoughts + 1))
    ax.set_yticklabels([f"Thought {i+1}" for i in range(num_thoughts)])
    ax.set_xlabel("Token Similarity")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot each thought's tokens
    for i, result in enumerate(results):
        y_pos = num_thoughts - i
        tokens = result['tokens']
        values = result['values']
        
        # Normalize values for visualization
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        # Plot bars for tokens
        for j, (token, val) in enumerate(zip(tokens, norm_values)):
            color = plt.cm.Blues(0.5 + 0.5 * val)
            ax.barh(y_pos, val, height=0.6, left=0, color=color, alpha=0.8)
            ax.text(val + 0.01, y_pos, f"{token} ({val:.2f})", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def visualize_attention(model, question, tokenizer, device, save_path=None):
    """
    Visualize attention patterns between continuous thoughts and input tokens.
    
    Args:
        model: The CODI model
        question: Question text
        tokenizer: Tokenizer
        device: Device for inference
        save_path: Path to save the visualization (if None, just shows the plot)
    """
    model.eval()
    
    # Encode the question
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Get attention weights (this requires modification to the forward pass to return attention weights)
    with torch.no_grad():
        # For this visualization, we would need to modify the model to return attention weights
        # as they're usually not directly accessible in the standard Hugging Face models
        # Here's a placeholder for what the visualization would look like:
        
        # Generate continuous thoughts and get their representations
        thought_representations, decoded_thoughts = model.decode_continuous_thoughts(
            inputs.input_ids, 
            inputs.attention_mask,
            top_k=1
        )
        
        # Get question tokens
        question_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Create placeholder attention matrix (random values for now)
        # In a real implementation, you would extract this from the model
        num_thoughts = len(thought_representations)
        num_tokens = len(question_tokens)
        attention_weights = np.random.random((num_thoughts, num_tokens))
        
        # Normalize weights for visualization
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(f"Attention Visualization\nQuestion: {question}", fontsize=14)
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set ticks and labels
    ax.set_yticks(range(num_thoughts))
    ax.set_yticklabels([f"Thought {i+1}: {tokenizer.convert_ids_to_tokens(decoded_thoughts[i][0][0])[0]}" 
                        for i in range(num_thoughts)])
    
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(question_tokens, rotation=45, ha='right')
    
    ax.set_xlabel('Question Tokens')
    ax.set_ylabel('Continuous Thoughts')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()