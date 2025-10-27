# TinyZero: A*PO for Language Model Reasoning

This repository contains a from-scratch PyTorch implementation of the **Advantage-based Policy Optimization (A*PO)** algorithm, used to fine-tune a small language model on mathematical reasoning tasks. This project, named "TinyZero," reproduces core concepts from the DeepSeek R1 paper by focusing on a multiplication task.

The entire training pipeline is designed to run on the `modal.com` cloud compute platform and adheres to a "minimalism" requirement, using only PyTorch for the core training and inference logic.

---
## How It Works: The A*PO Algorithm

Advantage-based Policy Optimization (A*PO) is a reinforcement learning algorithm designed to improve a language model's ability to generate high-quality responses. It works by comparing the model's generated answer to a pre-calculated "optimal" baseline, rewarding it for outperforming that baseline. The process is broken into two main stages:

### Stage 1: Offline V* Estimation
Before any training begins, we need to establish a high-quality baseline for each problem in our training set. This baseline is called the optimal value function, or $V^*(x)$.

1.  A frozen, pre-trained reference model generates multiple diverse potential solutions for each problem (e.g., 5 different answers for "What is 7 x 8?").
2.  Each generated solution is scored by a reward function (e.g., gets a reward of 1.0 if the answer is "56", 0.3 if it's close, and 0.0 otherwise).
3.  The rewards from all solutions are aggregated into a single, high-quality value, $V^*(x)$, which represents the "best achievable score" for that specific problem. This stage is computationally expensive but only needs to be done once per dataset.

### Stage 2: Online Policy Optimization
This is the main training loop where the policy model (the one we want to improve) learns.

1.  For a given problem, the policy model generates a single solution.
2.  This solution is scored by the same reward function.
3.  The **Advantage** is calculated. The advantage, $A(x,y)$, is the key to A\*PO. It measures how much better (or worse) the model's generated answer was compared to the pre-calculated $V^*(x)$ baseline. A small KL-divergence penalty term is included to prevent the policy from straying too far from the stable reference model.
4.  The model's weights are updated using this advantage. If the advantage is positive (the model did better than the baseline), the optimizer adjusts the weights to make that good answer more likely in the future. If the advantage is negative, it makes that bad answer less likely.

By repeating this process, the model learns to generate responses that consistently outperform the baseline, improving its reasoning and problem-solving skills.

---
## How to Replicate the Results

This project is configured to run on Modal. You will need a Modal account with a payment method for GPU usage.

### 1. Setup

First, clone the repository and set up the Python environment.

```bash
git clone <your-repo-url>
cd TinyZero-main
python -m venv venv
```

Activate the environment:
* On macOS/Linux: `source venv/bin/activate`
* On Windows: `.\venv\Scripts\activate`

Install the required packages:
```bash
pip install torch transformers accelerate numpy tqdm modal
```

### 2. Configure Modal
Set up your Modal account credentials on your local machine.
```bash
modal setup
```

### 3. Run Training
Execute the following command from your terminal to launch the training job on a Modal A100 GPU. This command replicates the experiment that achieved **30.60% accuracy**.

```bash
modal run modal_deploy.py --num-train 500 --num-test 50 --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --epochs 3 --batch-size 4
```

The script will build the environment, run the full A*PO pipeline, and save the final model to a persistent Modal Volume.

### 4. Download Results (Optional)
After the run completes, you can download the trained model and the generated `v_star_dict.json` file from the Modal volume to your local machine.

```bash
# Downloads the entire 'outputs' folder from the volume
modal volume get tinyzero-outputs /root/outputs .
```

---
## Final Experimental Results

The following results were obtained from the final successful training run, as specified in the command above.

### Configuration
* **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
* **Algorithm**: Advantage-based Policy Optimization (A*PO)
* **Task**: Multiplication of two random integers between 2 and 12.
* **Hardware**: 1x NVIDIA A100 GPU (via Modal)

### Hyperparameters
* **Training Samples**: 500
* **Test Samples**: 50
* **Epochs**: 3
* **Batch Size**: 4
* **Learning Rate**: `5e-6`
* **Precision**: `bfloat16`
* **A*PO Beta1 (V* Temp)**: `0.1`
* **A*PO Beta2 (KL Penalty)**: `0.01`
* **Samples for V\* Estimation**: 5 per prompt

### Performance
* **Final Test Accuracy**: **30.60%**
* **Total Training Time**: Approximately 2 hours (ran into the default `7200s` Modal timeout).
* **Evaluation Method**: The final accuracy is the average reward on the test set. The reward function provides `1.0` for a correct numerical answer, `0.3` for a close answer (within 50% error), and `0.0` otherwise.

---
## Code Structure

* `modal_deploy.py`: The main entrypoint for launching training jobs on Modal. It handles environment setup and remote execution.
* `train_local.py`: The core script that orchestrates data generation, model initialization, and the A*PO training pipeline.
* `training/apo_trainer.py`: A high-level class that manages the two stages of the A*PO algorithm.
* `training/stage1_v_estimation.py`: Contains the logic for the offline V* estimation phase.
* `training/stage2_policy_opt.py`: Implements the online policy optimization loop, including the A*PO loss function.

## CPU Experimental Results

### Test Run: Minimal CPU Test
- **Model:** Qwen/Qwen2.5-0.5B (500M parameters)
- **Device:** CPU (local)
- **Training samples:** 2
- **Test samples:** 1
- **Epochs:** 1

### Stage 1: V* Estimation
- **Time:** 13 seconds
- **V* Mean:** 0.9307
- **V* Range:** [0.9307, 0.9307]

### Stage 2: Policy Optimization
- **Time:** 51 seconds
- **Training Loss:** -28.4868
- **Training Reward:** 0.5000 (50% accuracy)

### Final Results
- **Test Accuracy:** 100.00% 
- **Total Time:** ~1 minute
- **Memory Usage:** ~2GB RAM
