#!/usr/bin/env python3
"""
测试数据格式和读取是否正常
"""

import pandas as pd
import os
from PIL import Image

def test_data_format():
    # 读取数据
    data_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"
    base_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItwImages/llavacot_test_images"
    
    print("📊 Loading data...")
    df = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"📏 Shape: {df.shape}")
    
    # 检查第一行数据
    print("\n🔍 First row data types:")
    first_row = df.iloc[0]
    for col in df.columns:
        value = first_row[col]
        print(f"  {col}: {type(value)} - {str(value)[:100]}...")
    
    # 测试_ensure_list函数
    def _ensure_list(data):
        """确保数据是列表格式"""
        import numpy as np
        
        if isinstance(data, list):
            return data
        elif isinstance(data, np.ndarray):
            # numpy数组转换为列表
            return data.tolist()
        elif isinstance(data, str):
            try:
                import ast
                return ast.literal_eval(data)
            except:
                return [data]
        else:
            return [str(data)]
    
    print("\n🧪 Testing data parsing:")
    test_row = df.iloc[0]
    
    # 测试各个字段
    for field in ['trp', 'trp_neg', 'tb']:
        if field in test_row:
            original = test_row[field]
            parsed = _ensure_list(original)
            print(f"  {field}:")
            print(f"    Original type: {type(original)}")
            print(f"    Parsed type: {type(parsed)}, length: {len(parsed)}")
            print(f"    First item: {parsed[0] if parsed else 'None'}")
    
    # 测试图像路径
    print("\n🖼️  Testing image loading:")
    test_image_path = test_row['image_path']
    full_path = os.path.join(base_path, test_image_path)
    print(f"  Image path: {test_image_path}")
    print(f"  Full path: {full_path}")
    print(f"  File exists: {os.path.exists(full_path)}")
    
    if os.path.exists(full_path):
        try:
            image = Image.open(full_path).convert('RGB')
            print(f"  ✅ Image loaded successfully: {image.size}")
        except Exception as e:
            print(f"  ❌ Failed to load image: {e}")
    
    # 测试前几个样本
    print(f"\n📋 Testing first 3 samples:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n  Sample {i}:")
        print(f"    ID: {row['id']}")
        print(f"    Image: {row['image_path']}")
        
        # 测试选项解析
        trp_list = _ensure_list(row['trp'])
        trp_neg_list = _ensure_list(row['trp_neg'])
        tb_list = _ensure_list(row['tb'])
        
        print(f"    TRP count: {len(trp_list)}")
        print(f"    TRP_NEG count: {len(trp_neg_list)}")
        print(f"    TB count: {len(tb_list)}")
        print(f"    Best TRP: {row['best_trp'][:50]}...")

if __name__ == "__main__":
    test_data_format() 