"""
modal_deploy.py
Modal 1.3+ compatible - Files read at runtime
"""
import modal
from pathlib import Path

app = modal.App("tinyzero-apo-deepseek")

# Create image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    )
)

volume = modal.Volume.from_name("tinyzero-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=28800,
    volumes={"/root/outputs": volume},
)
def train_on_modal(num_train, num_test, model_name, num_epochs, batch_size, code_files):
    """Run training with embedded code"""
    from pathlib import Path
    
    # Write code files to container
    Path("/root/training").mkdir(exist_ok=True)
    
    with open("/root/training/__init__.py", "w") as f:
        f.write('"""Training package"""')
    
    with open("/root/training/stage1_v_estimation.py", "w") as f:
        f.write(code_files["stage1"])
    
    with open("/root/training/stage2_policy_opt.py", "w") as f:
        f.write(code_files["stage2"])
    
    with open("/root/training/apo_trainer.py", "w") as f:
        f.write(code_files["apo_trainer"])
    
    with open("/root/train_local.py", "w") as f:
        f.write(code_files["train_local"])
    
    print("Code files written to container")
    
    # Run training
    import subprocess
    import sys
    
    sys.path.insert(0, "/root")
    
    cmd = [
        sys.executable,
        "/root/train_local.py",
        "--device", "cuda",
        "--num_train", str(num_train),
        "--num_test", str(num_test),
        "--model", model_name,
        "--epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", "5e-6",  
        "--num_samples_per_prompt", "5",
        "--output_dir", "/root/outputs"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        raise RuntimeError("Training failed")
    
    return {"status": "success", "model": model_name}


@app.local_entrypoint()
def main(
    num_train: int = 100,
    num_test: int = 20,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    epochs: int = 3,
    batch_size: int = 4,
):
    """Main entry point"""
    
    # Read files HERE (locally, before sending to Modal)
    def get_file_content(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    training_files = {
        "stage1": get_file_content("training/stage1_v_estimation.py"),
        "stage2": get_file_content("training/stage2_policy_opt.py"),
        "apo_trainer": get_file_content("training/apo_trainer.py"),
        "train_local": get_file_content("train_local.py"),
    }
    
    print("="*70)
    print("TinyZero A*PO on Modal.com CPU")
    print("="*70)
    print(f"Training: {num_train} samples")
    print(f"Test: {num_test} samples")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print("="*70)
    print("Cost: ~$0.75 (GPU - A100)")
    print("="*70)
    
    print("\nStarting training on Modal...")
    
    result = train_on_modal.remote(
        num_train, num_test, model, epochs, batch_size, training_files
    )
    
    print("\nCOMPLETE!")
    print(f"Status: {result['status']}")
    print("\nDownload: modal volume get tinyzero-outputs /root/outputs")