#!/usr/bin/env python3
"""
简单加载本地数据集的脚本
用法: python load_datasets.py --data_dir /path/to/datasets
"""
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="本地数据集根目录")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # 加载 WhatsUp
    print("Loading WhatsUp...")
    whatsup = load_dataset(f"{data_dir}/whats_up_vlms", trust_remote_code=True, split="test")
    print(f"WhatsUp: {len(whatsup)} samples")
    
    # 加载 VALSE
    print("Loading VALSE...")
    valse = load_dataset(f"{data_dir}/valse_vlms", trust_remote_code=True, split="test")
    print(f"VALSE: {len(valse)} samples")
    
    # 加载 CREPE
    print("Loading CREPE...")
    crepe = load_dataset(f"{data_dir}/crepe_vlms", trust_remote_code=True, split="test")
    print(f"CREPE: {len(crepe)} samples")
    
    # 加载 SugarCrepe
    print("Loading SugarCrepe...")
    sugarcrepe = load_dataset(f"{data_dir}/sugarcrepe_vlms", trust_remote_code=True, split="test")
    print(f"SugarCrepe: {len(sugarcrepe)} samples")
    
    # 加载 SugarCrepe++
    print("Loading SugarCrepe++...")
    sugarcrepepp = load_dataset(f"{data_dir}/sugarcrepepp_vlms", trust_remote_code=True, split="test")
    print(f"SugarCrepe++: {len(sugarcrepepp)} samples")
    
    print("\n所有数据集加载完成!")
    
    return whatsup, valse, crepe, sugarcrepe, sugarcrepepp

if __name__ == "__main__":
    main()
