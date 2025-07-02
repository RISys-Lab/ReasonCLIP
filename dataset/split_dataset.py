#!/usr/bin/env python3
"""
Split dataset script - 使用与训练代码相同的逻辑和随机种子分割数据集
将原始 parquet 文件分割成 train、eval、test 三个文件 (8:1:1)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def split_dataset():
    # 数据文件路径
    input_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet"
    output_dir = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Loading original dataset...")
    print(f"   Input file: {input_file}")
    
    # 读取原始数据
    df = pd.read_parquet(input_file)
    print(f"   Total samples: {len(df)}")
    
    # 验证数据格式（与训练代码相同的验证）
    required_columns = ["image_path", "tb", "trp"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"   Columns: {list(df.columns)}")
    print(f"   ✅ Data format validated")
    
    # 将DataFrame转换为字典格式（与训练代码保持一致）
    dataset_dict = df.to_dict('records')
    
    # 使用与训练代码完全相同的分割逻辑和随机种子
    print("\n📊 Splitting dataset with train_test_split (random_state=42)...")
    
    # 首先分出训练集和临时集 (8:2)
    train_data, temp_data = train_test_split(
        dataset_dict, 
        test_size=0.2,  # 20% 用于验证+测试
        random_state=42  # 与训练代码相同的随机种子
    )
    
    # 然后将临时集分为验证集和测试集 (1:1)
    eval_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,  # 50% 的临时数据作为测试集，50% 作为验证集
        random_state=42  # 与训练代码相同的随机种子
    )
    
    print(f"   - Train set: {len(train_data)} samples ({len(train_data)/len(df)*100:.1f}%)")
    print(f"   - Eval set:  {len(eval_data)} samples ({len(eval_data)/len(df)*100:.1f}%)")
    print(f"   - Test set:  {len(test_data)} samples ({len(test_data)/len(df)*100:.1f}%)")
    
    # 将字典格式转换回DataFrame
    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)
    test_df = pd.DataFrame(test_data)
    
    # 保存三个数据集
    train_file = os.path.join(output_dir, "llavacot_train.parquet")
    eval_file = os.path.join(output_dir, "llavacot_val.parquet")
    test_file = os.path.join(output_dir, "llavacot_test.parquet")
    
    print("\n💾 Saving split datasets...")
    
    train_df.to_parquet(train_file, index=False)
    print(f"   ✅ Train set saved: {train_file}")
    
    eval_df.to_parquet(eval_file, index=False)
    print(f"   ✅ Eval set saved: {eval_file}")
    
    test_df.to_parquet(test_file, index=False)
    print(f"   ✅ Test set saved: {test_file}")
    
    # 验证保存的文件
    print("\n🔍 Verification:")
    train_check = pd.read_parquet(train_file)
    eval_check = pd.read_parquet(eval_file)
    test_check = pd.read_parquet(test_file)
    
    print(f"   - Train file: {len(train_check)} samples")
    print(f"   - Eval file:  {len(eval_check)} samples")
    print(f"   - Test file:  {len(test_check)} samples")
    print(f"   - Total:      {len(train_check) + len(eval_check) + len(test_check)} samples")
    print(f"   - Original:   {len(df)} samples")
    
    if len(train_check) + len(eval_check) + len(test_check) == len(df):
        print("   ✅ All samples accounted for!")
    else:
        print("   ❌ Sample count mismatch!")
    
    print("\n🎉 Dataset splitting completed successfully!")
    print(f"📁 Output files:")
    print(f"   - {train_file}")
    print(f"   - {eval_file}")
    print(f"   - {test_file}")

if __name__ == "__main__":
    split_dataset() 