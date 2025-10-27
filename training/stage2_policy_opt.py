"""
training/stage2_policy_opt.py
Stage 2 of A*PO: Online Policy Optimization - FIXED FOR EXTREME LOSSES
Maximize: log π(y|x) · A(x,y)
where A(x,y) = r(x,y) + β₂·log(πref(y|x)/π(y|x)) - V*(x)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Callable


class APOPolicyOptimizer:
    """
    Correct A*PO Policy Optimization (Stage 2) with debugging
    """
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        v_star_dict: Dict[str, float],
        beta2: float = 0.01,
        learning_rate: float = 1e-5,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.v_star_dict = v_star_dict
        self.beta2 = beta2
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model.eval()
    
    def generate_single_response(self, prompt: str) -> tuple:
        """Generate single response for prompt"""
        was_training = self.model.training
        self.model.eval()
    
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
    
        if was_training:
            self.model.train()
    
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    
        return response_text.strip(), response_ids
    
    def compute_log_probs(self, prompt: str, response_ids: torch.Tensor, model) -> torch.Tensor:
        """
        Compute the mean log probability of a response given a prompt.
        This version is vectorized for efficiency and correctness.
        """
        response_length = len(response_ids)
        if response_length == 0:
            # If the model generates an empty response, the loss is zero.
            return torch.tensor(0.0, device=self.device, requires_grad=self.model.training)

        # 1. Tokenize the prompt to get its length.
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_len = prompt_tokens.input_ids.shape[1]

        # 2. Combine prompt and response to create the full sequence.
        full_ids = torch.cat([prompt_tokens.input_ids, response_ids.unsqueeze(0)], dim=1)
        
        # 3. Get the model's predictions (logits) for the entire sequence.
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(full_ids)
            logits = outputs.logits

        # 4. Isolate the logits that were used to predict the 'response' tokens.
        # To predict the first token of the response, we use the logit from the last token of the prompt.
        # So we slice from (prompt_len - 1) up to the token before the very last one.
        response_logits = logits[:, prompt_len - 1 : -1, :]

        # 5. Calculate the log probabilities of the actual tokens that were generated.
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # Use torch.gather to pick out the log_prob of each token in the response_ids sequence.
        # This is the vectorized equivalent of the old for loop.
        gathered_log_probs = torch.gather(
            log_probs, 2, response_ids.view(1, -1, 1)
        ).squeeze(-1)

        # 6. Return the MEAN log probability. THIS IS THE CRITICAL FIX.
        # Normalizing by the length prevents the loss from exploding.
        return gathered_log_probs.mean()

    def compute_apo_loss(
        self,
        prompt: str,
        response_text: str,
        response_ids: torch.Tensor,
        reward: float
    ) -> torch.Tensor:
        """
        Correct A*PO Loss:
        Maximize: log π(y|x) · A(x,y)
        where A(x,y) = r(x,y) + β₂·log(πref(y|x)/π(y|x)) - V*(x)
        """
        v_star = self.v_star_dict.get(prompt, 0.0)
        
        policy_log_prob = self.compute_log_probs(prompt, response_ids, self.model)
        
        with torch.no_grad():
            ref_log_prob = self.compute_log_probs(prompt, response_ids, self.ref_model)
        
        kl_term = ref_log_prob - policy_log_prob.detach()
        advantage = reward + self.beta2 * kl_term - v_star
        
        # Clip the advantage to prevent extreme values and stabilize training.
        advantage = torch.clamp(advantage, min=-10.0, max=10.0)

        # Loss: minimize -log π(y|x) · A(x,y)
        loss = -policy_log_prob * advantage
        
        return loss
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        reward_fn: Callable,
        epoch: int
    ) -> Dict:
        """Train one epoch with A*PO - WITH DEBUGGING"""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        num_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            prompts = batch['prompts']
            metadata = batch['metadata']
            
            batch_loss = 0.0
            batch_reward = 0.0
            
            for sample_idx, (prompt, meta) in enumerate(zip(prompts, metadata)):
                try:
                    print(f"\n[Batch {batch_idx+1}, Sample {sample_idx+1}/{len(prompts)}]")
                    print(f"  Prompt: {prompt[:60]}...")
                    
                    print("  → Generating response...")
                    response_text, response_ids = self.generate_single_response(prompt)
                    print(f"  → Response: {response_text[:60]}...")
                    print(f"  → Response length: {len(response_ids)} tokens")
                    
                    print("  → Computing reward...")
                    reward = reward_fn(response_text, meta)
                    print(f"  → Reward: {reward:.4f}")
                    
                    print("  → Computing A*PO loss...")
                    loss = self.compute_apo_loss(prompt, response_text, response_ids, reward)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: Invalid loss (NaN/inf), skipping")
                        batch_reward += reward
                        continue
                    
                    loss_value = abs(loss.item())
                    if loss_value > 50.0:
                        print(f"WARNING: Loss too extreme ({loss.item():.2f}), skipping")
                        batch_reward += reward
                        continue
                    
                    if loss_value < 1e-10:
                        print(f"WARNING: Loss is zero, skipping")
                        batch_reward += reward
                        continue
                    
                    print(f"  → Loss: {loss.item():.4f}")
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                    batch_loss += loss.item()
                    batch_reward += reward
                    num_samples += 1
                    
                    print(f"✓ Sample {sample_idx+1} complete!")
                    
                except KeyboardInterrupt:
                    print("\n\n Training interrupted by user (Ctrl+C)")
                    raise
                    
                except Exception as e:
                    print(f"\nERROR in Batch {batch_idx+1}, Sample {sample_idx+1}:")
                    print(f"   Error type: {type(e).__name__}")
                    print(f"   Error message: {e}")
                    
                    import traceback
                    print("\n   Full traceback:")
                    traceback.print_exc()
                    
                    print("\n   Skipping this sample and continuing...")
                    continue
            
            if len(prompts) > 0:
                avg_batch_loss = batch_loss / len(prompts) if num_samples > 0 else 0.0
                avg_batch_reward = batch_reward / len(prompts)
            else:
                avg_batch_loss = 0.0
                avg_batch_reward = 0.0
            
            total_loss += batch_loss
            total_reward += batch_reward
            
            pbar.set_postfix({
                'loss': f'{avg_batch_loss:.4f}',
                'reward': f'{avg_batch_reward:.4f}'
            })
        
        if num_samples > 0:
            return {
                'loss': total_loss / num_samples,
                'reward': total_reward / num_samples
            }
        else:
            return {
                'loss': 0.0,
                'reward': 0.0
            }