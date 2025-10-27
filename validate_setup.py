"""
validate_setup.py
Validates that your implementation meets all assignment requirements
Run this before any training to catch issues early
"""
import sys
from pathlib import Path
import re


def check_file_exists(filepath, description):
    """Check if required file exists"""
    if Path(filepath).exists():
        print(f" {description}: {filepath}")
        return True
    else:
        print(f" MISSING: {description}: {filepath}")
        return False


def check_code_pattern(filepath, pattern, description):
    """Check if code contains required pattern"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f" {description}")
            return True
        else:
            print(f" MISSING: {description}")
            return False
    except FileNotFoundError:
        print(f" File not found: {filepath}")
        return False


def main():
    print("="*70)
    print("TinyZero A*PO Implementation Validator")
    print("="*70)
    
    all_checks = []
    
    # 1. Check file structure
    print("\n Checking file structure...")
    all_checks.append(check_file_exists("train_local.py", "Single-process training script"))
    all_checks.append(check_file_exists("training/stage1_v_estimation.py", "Stage 1 V* estimation"))
    all_checks.append(check_file_exists("training/stage2_policy_opt.py", "Stage 2 policy optimization"))
    all_checks.append(check_file_exists("training/apo_trainer.py", "Main trainer"))
    all_checks.append(check_file_exists("modal_deploy.py", "Modal deployment"))
    all_checks.append(check_file_exists("requirements.txt", "Requirements"))
    all_checks.append(check_file_exists("README.md", "README"))
    
    # 2. Check A*PO implementation
    print("\n Checking A*PO implementation...")
    all_checks.append(check_code_pattern(
        "training/stage2_policy_opt.py",
        r"advantage.*=.*reward.*\+.*beta2.*\*.*kl_term.*-.*v_star",
        "Correct A*PO advantage formula"
    ))
    all_checks.append(check_code_pattern(
        "training/stage1_v_estimation.py",
        r"beta1.*\*.*log",
        "Correct V* estimation formula"
    ))
    
    # 3. Check FSDP support
    print("\n Checking FSDP support...")
    all_checks.append(check_code_pattern(
        "training/apo_trainer.py",
        r"from torch\.distributed\.fsdp import FullyShardedDataParallel",
        "FSDP import present"
    ))
    all_checks.append(check_code_pattern(
        "training/apo_trainer.py",
        r"use_fsdp",
        "FSDP parameter in trainer"
    ))
    
    # 4. Check no forbidden frameworks
    print("\n Checking for forbidden frameworks...")
    forbidden_patterns = [
        ("verl", "VERL framework"),
        ("deepspeed", "DeepSpeed framework"),
        ("from ray", "Ray framework"),
    ]
    
    for filepath in ["training/apo_trainer.py", "train_local.py"]:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            for pattern, name in forbidden_patterns:
                if pattern in content.lower():
                    print(f"✗ FORBIDDEN: {name} found in {filepath}")
                    all_checks.append(False)
                else:
                    all_checks.append(True)
        except FileNotFoundError:
            pass
    
    # 5. Check PyTorch only
    print("\n Checking PyTorch dependency...")
    all_checks.append(check_code_pattern(
        "requirements.txt",
        r"torch>=",
        "PyTorch in requirements"
    ))
    
    # 6. Check DeepSeek model
    print("\n Checking DeepSeek-R1 usage...")
    all_checks.append(check_code_pattern(
        "training/apo_trainer.py",
        r"deepseek-ai/DeepSeek-R1",
        "DeepSeek-R1 model reference"
    ))
    
    # 7. Check CPU compatibility
    print("\n Checking CPU compatibility...")
    all_checks.append(check_code_pattern(
        "train_local.py",
        r'device.*=.*"cpu"',
        "CPU device option"
    ))
    all_checks.append(check_code_pattern(
        "training/apo_trainer.py",
        r"torch\.float32",
        "FP32 support for CPU"
    ))
    
    # 8. Check single process
    print("\n Checking single-process design...")
    subprocess_patterns = [
        ("subprocess.Popen", "subprocess.Popen"),
        ("multiprocessing.Process", "multiprocessing.Process"),
        ("mp.Process", "multiprocessing Process"),
    ]
    
    single_process_ok = True
    for filepath in ["train_local.py", "training/apo_trainer.py"]:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            for pattern, name in subprocess_patterns:
                if pattern in content:
                    print(f"  WARNING: {name} found in {filepath}")
                    print(f"   (May violate single-process requirement)")
                    single_process_ok = False
        except FileNotFoundError:
            pass
    
    if single_process_ok:
        print("✓ No subprocess/multiprocessing calls found")
    all_checks.append(single_process_ok)
    
    # 9. Check Modal budget monitoring
    print("\n Checking Modal budget features...")
    all_checks.append(check_code_pattern(
        "modal_deploy.py",
        r"estimated_cost",
        "Cost estimation in Modal script"
    ))
    
    # 10. Check tasks (multiplication/countdown)
    print("\n Checking task implementation...")
    all_checks.append(check_code_pattern(
        "train_local.py",
        r"multiplication",
        "Multiplication task"
    ))
    
    # Summary
    print("\n" + "="*70)
    total_checks = len(all_checks)
    passed_checks = sum(all_checks)
    
    print(f"VALIDATION SUMMARY: {passed_checks}/{total_checks} checks passed")
    print("="*70)
    
    if passed_checks == total_checks:
        print("\n ALL CHECKS PASSED!")
        print("Your implementation meets all requirements.")
        print("\nNext steps:")
        print("1. Run: python train_local.py --device cpu --num_train 10 --num_test 5")
        print("2. Then: modal run modal_deploy.py --num-train 50 --num-test 10")
        return 0
    else:
        print(f"\n  {total_checks - passed_checks} ISSUES FOUND")
        print("Please fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())