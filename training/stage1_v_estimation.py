"""
training/stage1_v_estimation.py
Stage 1 of A*PO: Offline V* Estimation
V*(x) = β₁ * log(mean(exp(r/β₁)))
"""
import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm


class VStarEstimator:
    """
    Estimates optimal value function V* for each prompt
    V*(x) = β₁ * log(mean(exp(r/β₁)))
    """
    def __init__(
        self,
        model,
        tokenizer,
        num_samples_per_prompt: int = 4,
        beta1: float = 0.1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples_per_prompt
        self.beta1 = beta1
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
    
    def generate_responses(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate multiple responses for each prompt
        Batched for efficiency
        """
        all_responses = []
        
        for prompt in tqdm(prompts, desc="Generating responses for V*"):
            responses = []
            
            # Tokenize prompt once
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate multiple responses
            for _ in range(self.num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the generated part
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())
            
            all_responses.append(responses)
        
        return all_responses
    
    def compute_v_star(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute V*(x) = β₁ * log(mean(exp(r/β₁)))
        
        Args:
            rewards: Tensor of shape (num_prompts, num_samples_per_prompt)
        
        Returns:
            v_star: Tensor of shape (num_prompts,)
        """
        # Numerical stability: subtract max before exp
        max_rewards = rewards.max(dim=1, keepdim=True)[0]
        exp_rewards = torch.exp((rewards - max_rewards) / self.beta1)
        
        # Take mean across samples
        mean_exp_rewards = exp_rewards.mean(dim=1)
        
        # Take log and multiply by β₁, add back the max
        v_star = self.beta1 * torch.log(mean_exp_rewards) + max_rewards.squeeze()
        
        return v_star
    
    def estimate_v_star_for_dataset(
        self,
        dataloader,
        reward_fn
    ) -> Dict[str, float]:
        """
        Estimate V* for entire dataset
        """
        v_star_dict = {}
        
        print(f"\n{'='*70}")
        print(f"STAGE 1: V* ESTIMATION")
        print(f"Generating {self.num_samples} responses per prompt...")
        print(f"{'='*70}")
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            prompts = batch['prompts']
            metadata = batch['metadata']
            
            # Generate multiple responses per prompt
            all_responses = self.generate_responses(prompts)
            
            # Compute rewards for all responses
            batch_rewards = []
            for responses, meta in zip(all_responses, metadata):
                rewards = [reward_fn(response, meta) for response in responses]
                batch_rewards.append(rewards)
            
            # Convert to tensor: (batch_size, num_samples)
            rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
            
            # Compute V* for this batch
            v_star_values = self.compute_v_star(rewards_tensor)
            
            # Store in dictionary
            for prompt, v_star in zip(prompts, v_star_values):
                v_star_dict[prompt] = v_star.item()
        
        print(f"\nV* estimation complete for {len(v_star_dict)} prompts")
        if len(v_star_dict) > 0:
            print(f"V* statistics:")
            print(f"  Mean: {sum(v_star_dict.values()) / len(v_star_dict):.4f}")
            print(f"  Min:  {min(v_star_dict.values()):.4f}")
            print(f"  Max:  {max(v_star_dict.values()):.4f}")
        
        return v_star_dict


def save_v_star_dict(v_star_dict: Dict, save_path: str):
    """Save V* dictionary to file"""
    import json
    with open(save_path, 'w') as f:
        json.dump(v_star_dict, f, indent=2)
    print(f"Saved V* values to {save_path}")


def load_v_star_dict(load_path: str) -> Dict:
    """Load V* dictionary from file"""
    import json
    with open(load_path, 'r') as f:
        return json.load(f)