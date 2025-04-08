import torch
import numpy as np
from tqdm import tqdm

def train(model, dataloader, optimizer, scheduler, device, grad_accum_steps=1, epoch=0):
    """
    Train the model for one epoch.
    
    Args:
        model: The CODI model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        grad_accum_steps: Number of steps to accumulate gradients
        epoch: Current epoch number
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Setup progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch, return_dict=True)
        loss = outputs['loss']
        
        # Scale the loss for gradient accumulation
        loss = loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (step + 1) % grad_accum_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        total_loss += loss.item() * grad_accum_steps
        progress_bar.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'teacher_loss': outputs['teacher_loss'].item(),
            'student_loss': outputs['student_loss'].item(),
            'kd_loss': outputs['kd_loss'].item()
        })
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, is_test=False):
    """
    Evaluate the model on validation or test data.
    
    Args:
        model: The CODI model
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        is_test: Whether this is the test set evaluation
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluation" if not is_test else "Test")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get the targets (correct answers)
            answer_ids = batch['answer_input_ids']
            
            # Forward pass for evaluation
            outputs = model(batch, return_dict=True)
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Generate answers for accuracy calculation
            generated_ids = model.generate(
                batch['question_input_ids'],
                batch['question_attention_mask']
            )
            
            # Compare the generated answers with the ground truth
            # This is a simple exact match comparison
            # In practice, you might want a more sophisticated evaluation
            for i in range(generated_ids.size(0)):
                # Extract answer from generated text (depends on format)
                gen_answer = model.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                
                # Extract ground truth answer
                true_answer = model.tokenizer.decode(
                    batch['answer_input_ids'][i], 
                    skip_special_tokens=True
                ).split("The answer is:")[-1].strip()
                
                # Check if the answer is correct
                if true_answer in gen_answer:
                    correct += 1
                total += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'accuracy': correct / total if total > 0 else 0
            })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def decode_continuous_thoughts(model, question, tokenizer, device, top_k=5):
    """
    Decode and visualize the continuous thoughts for a given question.
    
    Args:
        model: The CODI model
        question: Question text
        tokenizer: Tokenizer for encoding
        device: Device to use for inference
        top_k: Number of top tokens to return for each thought
    
    Returns:
        Dictionary with thought representations and decoded tokens
    """
    model.eval()
    
    # Encode the question
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        # Get continuous thought representations and decoded tokens
        thought_representations, decoded_thoughts = model.decode_continuous_thoughts(
            inputs.input_ids, 
            inputs.attention_mask,
            top_k=top_k
        )
    
    # Process the results
    results = []
    for i, (indices, values) in enumerate(decoded_thoughts):
        # Get the top tokens for this thought
        tokens = tokenizer.convert_ids_to_tokens(indices[0])
        # Get the corresponding values (similarities)
        token_values = values[0].cpu().numpy()
        
        results.append({
            'thought_idx': i,
            'tokens': tokens,
            'values': token_values,
            'representation': thought_representations[i][0].cpu().numpy()
        })
    
    return results