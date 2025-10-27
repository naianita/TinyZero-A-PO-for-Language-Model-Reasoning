"""
train_local.py
Single-process CPU/GPU training script for TinyZero A*PO
Follows minimalism requirement - everything in one process

CREATE THIS FILE IN THE ROOT DIRECTORY OF YOUR PROJECT
"""
import torch
from pathlib import Path
import argparse
import sys
import json

# Add paths - this allows importing from training/
sys.path.insert(0, str(Path(__file__).parent))

from torch.utils.data import DataLoader, Dataset


# ============================================================================
# DATASET CLASS
# ============================================================================

class TinyZeroDataset(Dataset):
    """Simple dataset for multiplication tasks"""
    def __init__(self, data_path: str, tokenizer=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item['prompt'],
            'task_type': item['task_type'],
            'metadata': item
        }
    
    def collate_fn(self, batch):
        prompts = [item['prompt'] for item in batch]
        task_types = [item['task_type'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        return {
            'prompts': prompts,
            'task_types': task_types,
            'metadata': metadata
        }


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_multiplication_dataset(num_samples: int, output_path: str):
    """Generate multiplication tasks"""
    import random
    
    dataset = []
    for _ in range(num_samples):
        # Easy problems for CPU testing
        num1 = random.randint(2, 12)
        num2 = random.randint(2, 12)
        
        dataset.append({
            "task_type": "multiplication",
            "num1": num1,
            "num2": num2,
            "answer": num1 * num2,
            "prompt": f"What is {num1} × {num2}? Show your reasoning step by step."
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f" Created {len(dataset)} multiplication samples -> {output_path}")


def generate_countdown_dataset(num_samples: int, output_path: str):
    """Generate countdown tasks"""
    import random
    
    dataset = []
    for _ in range(num_samples):
        target = random.randint(50, 100)
        numbers = random.sample(range(1, 25), k=6)
        
        dataset.append({
            "task_type": "countdown",
            "target": target,
            "numbers": numbers,
            "prompt": f"Use the numbers {numbers} to reach {target}. You can use +, -, *, / operations. Show your work."
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f" Created {len(dataset)} countdown samples -> {output_path}")


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def extract_answer(solution: str) -> int:
    """Extract numerical answer from solution"""
    import re
    numbers = re.findall(r'\d+', solution)
    return int(numbers[-1]) if numbers else -1


def compute_reward(solution: str, metadata: dict) -> float:
    """
    Compute reward for multiplication - WITH PARTIAL CREDIT
    """
    task_type = metadata.get('task_type', 'multiplication')
    
    if task_type == 'multiplication':
        correct_answer = metadata['answer']
        extracted = extract_answer(solution)
        
        # Exact match
        if extracted == correct_answer:
            return 1.0
        
        # PARTIAL CREDIT: Close answers get partial reward
        if extracted > 0:
            error = abs(extracted - correct_answer) / max(correct_answer, 1)
            if error < 0.5:  # Within 50% of correct answer
                return 0.3  # Partial credit
        
        return 0.0


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TinyZero A*PO Training - Single Process")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cpu", 
                       choices=["cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--use_fsdp", action="store_true", 
                       help="Use FSDP (requires GPU and distributed)")
    
    # Data settings
    parser.add_argument("--task", type=str, default="multiplication",
                       choices=["multiplication", "countdown", "both"],
                       help="Which task to train on")
    parser.add_argument("--num_train", type=int, default=10, 
                       help="Number of training samples")
    parser.add_argument("--num_test", type=int, default=5, 
                       help="Number of test samples")
    
    # Model settings
    parser.add_argument("--model", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                       help="Model to use")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4,
                       help="Number of samples for V* estimation")
    
    # Other settings
    parser.add_argument("--skip_stage1", action="store_true", 
                       help="Skip V* estimation (load from cache)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("TinyZero A*PO Training - Single Process")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"FSDP: {args.use_fsdp}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Training samples: {args.num_train}")
    print(f"Test samples: {args.num_test}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    # Validate FSDP settings
    if args.use_fsdp:
        if args.device == "cpu":
            print("\n ERROR: FSDP requires CUDA, cannot use with CPU")
            print("Remove --use_fsdp flag or use --device cuda")
            return
        
        import torch.distributed as dist
        if not dist.is_initialized():
            print("\nInitializing distributed for FSDP...")
            dist.init_process_group(backend="nccl")
            print(" Distributed initialized")
    
    # Generate datasets
    print("\n Generating datasets...")
    Path("data").mkdir(exist_ok=True)
    
    if args.task in ["multiplication", "both"]:
        generate_multiplication_dataset(args.num_train, "data/mult_train.json")
        generate_multiplication_dataset(args.num_test, "data/mult_test.json")
        train_data_path = "data/mult_train.json"
        test_data_path = "data/mult_test.json"
    
    if args.task in ["countdown", "both"]:
        generate_countdown_dataset(args.num_train, "data/countdown_train.json")
        generate_countdown_dataset(args.num_test, "data/countdown_test.json")
        if args.task == "countdown":
            train_data_path = "data/countdown_train.json"
            test_data_path = "data/countdown_test.json"
    
    # Import trainer (all in same process)
    print("\n Initializing trainer...")
    from training.apo_trainer import APOTrainer
    
    # Initialize trainer128
    trainer = APOTrainer(
        model_name=args.model,
        device=args.device,
        use_fsdp=args.use_fsdp,
        num_samples_per_prompt=args.num_samples_per_prompt,
        beta1=0.1,
        beta2=0.01,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_new_tokens=50,
        temperature=1.5,
        output_dir=args.output_dir
    )
    
    # Load datasets
    print("\n Loading datasets...")
    train_dataset = TinyZeroDataset(train_data_path, trainer.tokenizer)
    test_dataset = TinyZeroDataset(test_data_path, trainer.tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    print(f" Train samples: {len(train_dataset)}")
    print(f" Test samples: {len(test_dataset)}")
    
    # Train
    print("\n Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        reward_fn=compute_reward,
        skip_stage1=args.skip_stage1
    )
    
    # Evaluate
    print("\n Evaluating on test set...")
    accuracy = trainer.evaluate(test_loader, compute_reward)
    
    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_model(final_path)
    
    # Print summary
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to: {final_path}")
    print(f"V* dict saved to: {Path(args.output_dir) / 'v_star_dict.json'}")
    print("="*70)
    
    # Cleanup FSDP if used
    if args.use_fsdp:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()