"""
test_cpu_minimal.py
Minimal CPU test with a TINY model that will actually complete
"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_local import *

def main():
    print("="*70)
    print("MINIMAL CPU TEST - Using Tiny Model")
    print("="*70)
    
    # Generate tiny dataset
    Path("data").mkdir(exist_ok=True)
    generate_multiplication_dataset(2, "data/mult_train_tiny.json")
    generate_multiplication_dataset(1, "data/mult_test_tiny.json")
    print(" Generated minimal dataset")
    
    from training.apo_trainer import APOTrainer
    from torch.utils.data import DataLoader
    
    # Use TINY model (500M params instead of 1.5B)
    print("\n Loading TINY model (500M params)...")
    trainer = APOTrainer(
        model_name="Qwen/Qwen2.5-0.5B",  # Much smaller!
        device="cpu",
        use_fsdp=False,
        num_samples_per_prompt=2,  # Minimal
        beta1=0.1,
        beta2=0.01,
        learning_rate=1e-5,
        num_epochs=1,
        batch_size=1,
        max_new_tokens=50,  # Shorter responses
        temperature=1.0,
        output_dir="outputs_tiny"
    )
    
    print(" Model loaded")
    
    # Load datasets
    train_dataset = TinyZeroDataset("data/mult_train_tiny.json", trainer.tokenizer)
    test_dataset = TinyZeroDataset("data/mult_test_tiny.json", trainer.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Train
    print("\n Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        reward_fn=compute_reward,
        skip_stage1=False
    )
    
    # Evaluate
    print("\n Evaluating...")
    accuracy = trainer.evaluate(test_loader, compute_reward)
    
    # Save
    trainer.save_model("outputs_tiny/final_model")
    
    print("\n" + "="*70)
    print(" MINIMAL TEST COMPLETE!")
    print("="*70)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()