#!/usr/bin/env python3
"""
示例脚本：展示如何使用不同模型进行Zero-Shot Reasoning Classification评估

使用方法：
python eval/run_evaluation_examples.py
"""

import subprocess
import sys
import os

def run_evaluation(model_type, model_name, task, output_suffix=""):
    """运行单个任务评估"""
    print(f"\n{'='*60}")
    print(f"Running {task} evaluation with {model_type.upper()}: {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "eval/retrieval_reasonpro.py",
        "--model_type", model_type,
        "--model_name", model_name,
        "--task", task,
        "--output_dir", f"/home/muzammal/Projects/CLIP-R/eval/results_{model_type}{output_suffix}",
        "--device", "auto"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd="/home/muzammal/Projects/CLIP-R")
        print(f"✅ {model_type.upper()} {model_name} {task} evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_type.upper()} {model_name} {task} evaluation failed: {e}")
    except Exception as e:
        print(f"❌ Error running {model_type.upper()} {model_name} {task}: {e}")

def main():
    print("🚀 Starting Multi-Model Zero-Shot Reasoning Classification Evaluation")
    
    # 定义要测试的模型
    models_to_test = [
        # CLIP models (Hugging Face implementation)
        # ("clip", "openai/clip-vit-base-patch32"),
        # ("clip", "openai/clip-vit-base-patch16"),
        ("clip", "openai/clip-vit-large-patch14"),
        ("clip", "fesvhtr/clip_r_best_model_demo_0621_192211"),
        
        # OpenCLIP models
        # ("openclip", "ViT-B-32"),
        # ("openclip", "ViT-L-14"),
        
        # # SigLIP models
        # ("siglip", "google/siglip-base-patch16-224"),
        # ("siglip", "google/siglip-large-patch16-256"),
    ]
    
    tasks_to_test = [
        "logic_val",
        "best_reason", 
        "reason_id"
    ]
    
    print(f"📊 Testing {len(models_to_test)} model configurations with {len(tasks_to_test)} tasks each...")
    
    # 运行评估
    for model_type, model_name in models_to_test:
        for task in tasks_to_test:
            run_evaluation(model_type, model_name, task)
    
    print(f"\n{'='*60}")
    print("🎉 All evaluations completed!")
    print("📊 Check the results in the respective output directories:")
    for model_type, model_name in models_to_test:
        print(f"   - ./results_{model_type}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 