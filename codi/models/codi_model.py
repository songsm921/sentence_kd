import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

class CODIProjection(nn.Module):
    """
    Projection layer for continuous thoughts based on the CODI paper.
    Transforms hidden states for continuous thought tokens.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        return self.projection(x)

class CODI(nn.Module):
    def __init__(
        self,
        model_name,
        num_latent=6,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        bot_token_id=None,
        eot_token_id=None,
        tokenizer=None,
    ):
        super().__init__()
        
        # Initialize model parameters
        self.num_latent = num_latent
        self.alpha = alpha  # Weight for teacher CE loss
        self.beta = beta    # Weight for student CE loss
        self.gamma = gamma  # Weight for knowledge distillation loss
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.tokenizer = tokenizer
        
        # Load the shared model for both teacher and student roles
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 1024
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map ='auto')
        
        # Resize the token embeddings if we added special tokens
        if tokenizer is not None and len(tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(tokenizer))
            print('Resized')
        
        # Get the hidden dimension from the model configuration
        self.hidden_dim = self.model.config.hidden_size
        
        # Initialize the projection layer for continuous thought transformation
        self.projection = CODIProjection(self.hidden_dim)

    def forward(self, inputs, return_dict=False):
        """
        Forward pass for CODI model.
        """
        device = inputs['question_input_ids'].device
        batch_size = inputs['question_input_ids'].shape[0]
        
        # Extract target positions for knowledge distillation
        teacher_target_position = inputs['teacher_target_position']
        student_target_position = inputs['student_target_position']
        
        # ------------------------------
        # Teacher Task (explicit CoT)
        # ------------------------------
        teacher_outputs = self.model(
            input_ids=inputs['cot_answer_input_ids'],
            attention_mask=inputs['cot_answer_attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Teacher CE Loss
        teacher_logits = teacher_outputs.logits[:, :-1, :]
        teacher_targets = inputs['cot_answer_input_ids'][:, 1:]
        teacher_loss_mask = inputs['cot_answer_attention_mask'][:, 1:]
        
        teacher_ce_loss = F.cross_entropy(
            teacher_logits.reshape(-1, teacher_logits.size(-1)),
            teacher_targets.reshape(-1),
            reduction='none'
        )
        teacher_ce_loss = (teacher_ce_loss * teacher_loss_mask.reshape(-1)).sum() / teacher_loss_mask.sum()
        
        # ------------------------------
        # Student Task (continuous CoT)
        # ------------------------------
        # Get question representation
        question_outputs = self.model(
            input_ids=inputs['question_input_ids'],
            attention_mask=inputs['question_attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the hidden state of the last token in the question
        last_hidden = question_outputs.hidden_states[-1][:, -1, :]
        
        # Initial hidden state for continuous thoughts
        latent = self.projection(last_hidden)
        
        # Generate past key values for incremental decoding
        past_key_values = None
        
        # Autoregressive generation of continuous thoughts
        for i in range(self.num_latent):
            # Process one continuous thought token at a time
            # attention_mask = torch.ones(batch_size, 1, device=device)
            # if past_key_values is not None:
            #     # past_key_values의 길이에 맞게 attention_mask 확장
            #     seq_len = past_key_values[0][0].shape[2] + 1  # 현재 position + past
            #     extended_mask = torch.ones(batch_size, seq_len, device=device)
            #     attention_mask = extended_mask
            continuous_outputs = self.model(
                inputs_embeds=latent.unsqueeze(1),
                past_key_values=past_key_values,
                # attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Update past key values for efficient decoding
            past_key_values = continuous_outputs.past_key_values
            
            # Get the new hidden state and apply projection
            latent = continuous_outputs.hidden_states[-1][:, -1, :]
            latent = self.projection(latent)
        
        # Generate answer after continuous thoughts
        if past_key_values is not None:
            # 기존 past_key_values의 시퀀스 길이 확인
            past_length = past_key_values[0][0].shape[2]  # key의 시퀀스 길이
            
            # 새로운 attention mask 생성
            # 과거 토큰들(past_length)에 대해서는 모두 1, 현재 입력 토큰에 대해서는 원래 마스크 사용
            extended_attention_mask = torch.ones(
                (batch_size, past_length), 
                dtype=torch.long, 
                device=device
            )
            
            # 원래 마스크와 연결
            combined_attention_mask = torch.cat(
                [extended_attention_mask, inputs['answer_attention_mask']], 
                dim=1
            )
            
            answer_outputs = self.model(
                input_ids=inputs['answer_input_ids'],
                attention_mask=combined_attention_mask,  # 확장된 마스크 사용
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            answer_outputs = self.model(
                input_ids=inputs['answer_input_ids'],
                attention_mask=inputs['answer_attention_mask'],
                past_key_values=None,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Student CE Loss
        student_logits = answer_outputs.logits[:, :-1, :]
        student_targets = inputs['answer_input_ids'][:, 1:]
        student_loss_mask = inputs['answer_attention_mask'][:, 1:]
        
        student_ce_loss = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            student_targets.reshape(-1),
            reduction='none'
        )
        student_ce_loss = (student_ce_loss * student_loss_mask.reshape(-1)).sum() / student_loss_mask.sum()
        
        # ------------------------------
        # Knowledge Distillation Loss (원본 코드와 더 유사하게 수정)
        # ------------------------------
        batch_indices = torch.arange(batch_size, device=device)
        
        teacher_hidden_states = []
        for layer_idx in range(len(teacher_outputs.hidden_states)):
            teacher_layer_hidden = teacher_outputs.hidden_states[layer_idx]
            # Gather the specific token position for each item in the batch
            idx = torch.stack([batch_indices, teacher_target_position], dim=1)
            teacher_layer_target = teacher_layer_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))[:, 0]
            teacher_hidden_states.append(teacher_layer_target)
        
        # Extract target student hidden states for distillation
        student_hidden_states = []
        for layer_idx in range(len(answer_outputs.hidden_states)):
            student_layer_hidden = answer_outputs.hidden_states[layer_idx]
            # Gather the specific token position for each item in the batch
            idx = torch.stack([batch_indices, student_target_position], dim=1)
            student_layer_target = student_layer_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))[:, 0]
            student_hidden_states.append(student_layer_target)

        
        kd_loss = 0
        for layer_idx in range(len(teacher_hidden_states)):
            # Normalize by teacher's std as mentioned in the paper
            teacher_std = teacher_hidden_states[layer_idx].std(dim=1, keepdim=True) # ????
            # print(teacher_hidden_states[layer_idx].shape)
            normalized_teacher = teacher_hidden_states[layer_idx] / (teacher_std + 1e-6)
            normalized_student = student_hidden_states[layer_idx] / (teacher_std + 1e-6)
            
            # Stop gradient for teacher as mentioned in the paper
            layer_kd_loss = F.l1_loss(normalized_student, normalized_teacher.detach())
            kd_loss += layer_kd_loss
        
        # Average across layers
        kd_loss = kd_loss / len(teacher_hidden_states)
        kd_loss = kd_loss / len(teacher_outputs.hidden_states)
        
        # 최종 손실 계산
        total_loss = self.alpha * teacher_ce_loss + self.beta * student_ce_loss + self.gamma * kd_loss
        
        if return_dict:
            return {
                'loss': total_loss,
                'teacher_loss': teacher_ce_loss,
                'student_loss': student_ce_loss,
                'kd_loss': kd_loss,
            }
        
        return total_loss
    
    def generate(self, question_input_ids, question_attention_mask, max_answer_length=20):
        """
        Generate answers using continuous thoughts during inference.
        """
        device = question_input_ids.device
        batch_size = question_input_ids.shape[0]
        
        # Get question representation
        question_outputs = self.model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the hidden state of the last token in the question
        last_hidden = question_outputs.hidden_states[-1][:, -1, :]
        
        # Initial hidden state for continuous thoughts
        latent = self.projection(last_hidden)
        
        # Generate past key values for incremental decoding
        past_key_values = None
        
        # Autoregressive generation of continuous thoughts
        for i in range(self.num_latent):
            # Process one continuous thought token at a time
            continuous_outputs = self.model(
                inputs_embeds=latent.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Update past key values for efficient decoding
            past_key_values = continuous_outputs.past_key_values
            
            # Get the new hidden state and apply projection
            latent = continuous_outputs.hidden_states[-1][:, -1, :]
            latent = self.projection(latent)
        
        # Add <eot> token to signal the end of continuous thoughts
        eot_input_ids = torch.tensor([[self.eot_token_id]], device=device).repeat(batch_size, 1)
        
        # Generate the answer using the model's generate method
        eot_outputs = self.model(
            input_ids=eot_input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = eot_outputs.past_key_values
        
        # Generate answer tokens autoregressively
        generated_ids = []
        curr_input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device).repeat(batch_size, 1)
        
        for _ in range(max_answer_length):
            outputs = self.model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token)
            curr_input_ids = next_token
            
            # Stop if all sequences have generated the EOS token
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        # Concatenate all tokens
        generated_ids = torch.cat(generated_ids, dim=1)
        
        return generated_ids
    
    def decode_continuous_thoughts(self, question_input_ids, question_attention_mask, top_k=5):
        """
        Decode continuous thoughts into vocabulary space for interpretability.
        Returns top-k tokens for each continuous thought.
        """
        device = question_input_ids.device
        batch_size = question_input_ids.shape[0]
        
        # Get question representation
        question_outputs = self.model(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the hidden state of the last token in the question
        last_hidden = question_outputs.hidden_states[-1][:, -1, :]
        
        # Initial hidden state for continuous thoughts
        latent = self.projection(last_hidden)
        
        # Generate past key values for incremental decoding
        past_key_values = None
        
        # Store thought representations and decoded tokens
        thought_representations = []
        decoded_thoughts = []
        
        # Get the token embeddings for projection into vocabulary space
        token_embeddings = self.model.get_input_embeddings().weight
        
        # Autoregressive generation of continuous thoughts
        for i in range(self.num_latent):
            # Process one continuous thought token at a time
            continuous_outputs = self.model(
                inputs_embeds=latent.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Update past key values for efficient decoding
            past_key_values = continuous_outputs.past_key_values
            
            # Get the new hidden state and apply projection
            latent = continuous_outputs.hidden_states[-1][:, -1, :]
            thought_representations.append(latent.clone())
            latent = self.projection(latent)
            
            # Project into vocabulary space
            # Compute similarity with all token embeddings
            similarity = torch.matmul(latent, token_embeddings.transpose(0, 1))
            
            # Get top-k tokens
            top_values, top_indices = torch.topk(similarity, top_k, dim=-1)
            
            decoded_thoughts.append((top_indices, top_values))
        
        return thought_representations, decoded_thoughts