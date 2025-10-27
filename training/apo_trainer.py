"""
training/apo_trainer.py
Complete A*PO Training Pipeline with FSDP Support
Uses DeepSeek-R1 models
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import json
from pathlib import Path

from .stage1_v_estimation import VStarEstimator, save_v_star_dict, load_v_star_dict
from .stage2_policy_opt import APOPolicyOptimizer


class APOTrainer:
    """
    Complete A*PO training pipeline with FSDP support
    """
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str = "cpu",
        use_fsdp: bool = False,
        # Stage 1 parameters
        num_samples_per_prompt: int = 4,
        beta1: float = 0.1,
        # Stage 2 parameters
        beta2: float = 0.01,
        learning_rate: float = 1e-5,
        num_epochs: int = 2,
        batch_size: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        # Paths
        output_dir: str = "outputs",
        v_star_cache_path: Optional[str] = None
    ):
        """Initialize A*PO trainer with DeepSeek-R1"""
        self.device = device
        self.use_fsdp = use_fsdp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading model: {model_name}")
        print(f"Device: {device}")
        print(f"FSDP: {use_fsdp}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype based on device
        if device == "cpu":
            dtype = torch.float32
            device_map = None
        else:
            dtype = torch.bfloat16 
            device_map = "auto" if not use_fsdp else None
        
        # Load policy model (to be trained)
        print("Loading policy model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing to save memory
        '''if device == "cuda":
            self.model.gradient_checkpointing_enable()'''
        
        if device == "cpu" and device_map is None:
            self.model = self.model.to(device)
        
        # Load reference model (frozen)
        print("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu" and device_map is None:
            self.ref_model = self.ref_model.to(device)
        
        # Wrap with FSDP if requested
        if use_fsdp and torch.cuda.is_available():
            print("Wrapping models with FSDP...")
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.model = FSDP(self.model)
            self.ref_model = FSDP(self.ref_model)
        
        # Store parameters
        self.num_samples_per_prompt = num_samples_per_prompt
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.v_star_cache_path = v_star_cache_path
        
        self.v_star_dict = None
        
        print(f"✓ Model loaded successfully")
    
    def stage1_estimate_vstar(
        self,
        train_dataloader: DataLoader,
        reward_fn,
        force_recompute: bool = False
    ):
        """Stage 1: Estimate V* for all training prompts"""
        # Check cache
        if self.v_star_cache_path and Path(self.v_star_cache_path).exists() and not force_recompute:
            print(f"\nLoading V* from cache: {self.v_star_cache_path}")
            self.v_star_dict = load_v_star_dict(self.v_star_cache_path)
            return
        
        print("\n" + "="*70)
        print("STAGE 1: V* ESTIMATION")
        print("="*70)
        
        # Create V* estimator
        estimator = VStarEstimator(
            model=self.ref_model,
            tokenizer=self.tokenizer,
            num_samples_per_prompt=self.num_samples_per_prompt,
            beta1=self.beta1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
        
        # Estimate V*
        self.v_star_dict = estimator.estimate_v_star_for_dataset(
            train_dataloader,
            reward_fn
        )
        
        # Save V* dictionary
        v_star_path = self.output_dir / "v_star_dict.json"
        save_v_star_dict(self.v_star_dict, str(v_star_path))
    
    def stage2_optimize_policy(
        self,
        train_dataloader: DataLoader,
        reward_fn
    ):
        """Stage 2: Optimize policy using V* values"""
        print("\n" + "="*70)
        print("STAGE 2: POLICY OPTIMIZATION")
        print("="*70)
        
        # Create policy optimizer
        optimizer = APOPolicyOptimizer(
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            v_star_dict=self.v_star_dict,
            beta2=self.beta2,
            learning_rate=self.learning_rate,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            stats = optimizer.train_epoch(
                train_dataloader,
                reward_fn,
                epoch + 1
            )
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Loss:   {stats['loss']:.4f}")
            print(f"  Reward: {stats['reward']:.4f}")
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch + 1}"
            self.save_model(checkpoint_path)
    
    def train(
        self,
        train_dataloader: DataLoader,
        reward_fn,
        skip_stage1: bool = False
    ):
        """Complete A*PO training pipeline"""
        print("\n" + "="*70)
        print("A*PO TRAINING PIPELINE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"FSDP: {self.use_fsdp}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Beta1 (V* temp): {self.beta1}")
        print(f"Beta2 (KL penalty): {self.beta2}")
        print("="*70)
        
        # Stage 1: V* Estimation
        if not skip_stage1:
            self.stage1_estimate_vstar(train_dataloader, reward_fn)
        else:
            print("\nSkipping Stage 1 (loading V* from cache)")
            if self.v_star_cache_path:
                self.v_star_dict = load_v_star_dict(self.v_star_cache_path)
        
        # Stage 2: Policy Optimization
        self.stage2_optimize_policy(train_dataloader, reward_fn)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
    
    def save_model(self, path):
        """Save trained model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Unwrap FSDP if needed
        model_to_save = self.model
        if self.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model_to_save, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model_to_save.state_dict()
                torch.save(state_dict, Path(path) / "model.pt")
        else:
            model_to_save.save_pretrained(path)
        
        self.tokenizer.save_pretrained(path)
        print(f"✓ Model saved to {path}")
    
    def evaluate(self, test_dataloader: DataLoader, reward_fn):
        """Evaluate model on test set"""
        self.model.eval()
        total_reward = 0.0
        num_samples = 0
        
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        with torch.no_grad():
            for batch in test_dataloader:
                prompts = batch['prompts']
                metadata = batch['metadata']
                
                for prompt, meta in zip(prompts, metadata):
                    # Generate response
                    inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Compute reward
                    reward = reward_fn(response, meta)
                    total_reward += reward
                    num_samples += 1
        
        avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
        print(f"\nTest Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Accuracy: {avg_reward * 100:.2f}%")
        print("="*70)
        
        return avg_reward