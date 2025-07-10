import pandas as pd
from tqdm import tqdm
import re
import os

reason_itw = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_merged.parquet"
df1 = pd.read_parquet(reason_itw)

print(df1.head())
print(df1.columns)
print(f"数据形状: {df1.shape}")

print("\n" + "="*50)
print("创建options列")
print("="*50)

def create_options(row):
    """
    按照 best_trp, trp(移除best_trp), trp_neg, tb 的顺序组合选项
    """
    best_trp = row['best_trp']
    trp = row['trp']
    trp_neg = row['trp_neg'] 
    tb = row['tb']
    
    # 从trp中移除best_trp
    trp_filtered = [item for item in trp if item != best_trp]
    
    # 按顺序组合：best_trp + trp(without best_trp) + trp_neg + tb
    options = [best_trp] + trp_filtered + list(trp_neg) + list(tb)
    
    return options

# 创建options列
print("正在创建options列...")
df1['options'] = df1.apply(create_options, axis=1)

print(f"创建后数据形状: {df1.shape}")
print("新的列结构:")
print(df1.columns.tolist())

# 检查options列的大小
options_sizes = df1['options'].apply(len)
print(f"\noptions列大小统计: {options_sizes.value_counts().sort_index()}")

# 显示几个例子
print("\n前3行的options示例:")
for i in range(3):
    print(f"行{i}:")
    print(f"  best_trp: {df1.iloc[i]['best_trp']}")
    print(f"  trp: {list(df1.iloc[i]['trp'])}")
    print(f"  trp_neg: {list(df1.iloc[i]['trp_neg'])}")
    print(f"  tb: {list(df1.iloc[i]['tb'])}")
    print(f"  options: {df1.iloc[i]['options']}")
    print(f"  options长度: {len(df1.iloc[i]['options'])}")
    print()

# 保存包含options列的新文件
output_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"
df1.to_parquet(output_path, index=False)
print(f"✅ 已保存包含options列的数据到: {output_path}")
print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


