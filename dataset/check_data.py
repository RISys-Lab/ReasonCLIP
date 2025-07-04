import pandas as pd
from tqdm import tqdm
import re

reason1 = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonLite100M/cc12m_tb/chunk_00/cc12m_visual_ckpt_0040000_20250625_192255/14_000000_000000.parquet"
reason2 = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonLite100M/cc12m_tb/chunk_00/cc12m_visual_ckpt_0080000_20250626_004704/27_000000_000000.parquet"
df1 = pd.read_parquet(reason1)
df2 = pd.read_parquet(reason2)

print("=== 基本信息 ===")
print(f"df1 columns: {df1.columns.tolist()}")
print(f"df2 columns: {df2.columns.tolist()}")
print(f"df1 length: {len(df1)}")
print(f"df2 length: {len(df2)}")

print("\n=== ID重复检查 ===")

# 检查是否有id列
if 'id' in df1.columns and 'id' in df2.columns:
    # 检查df1内部重复
    df1_duplicates = df1[df1.duplicated(subset=['id'], keep=False)]
    print(f"df1内部重复的id数量: {len(df1_duplicates)}")
    if len(df1_duplicates) > 0:
        print(f"df1重复的唯一id数量: {df1_duplicates['id'].nunique()}")
        if df1_duplicates['id'].nunique() <= 10:
            print(f"df1重复的id: {df1_duplicates['id'].unique().tolist()}")
    
    # 检查df2内部重复
    df2_duplicates = df2[df2.duplicated(subset=['id'], keep=False)]
    print(f"df2内部重复的id数量: {len(df2_duplicates)}")
    if len(df2_duplicates) > 0:
        print(f"df2重复的唯一id数量: {df2_duplicates['id'].nunique()}")
        if df2_duplicates['id'].nunique() <= 10:
            print(f"df2重复的id: {df2_duplicates['id'].unique().tolist()}")
    
    # 检查两个数据框之间的重复
    df1_ids = set(df1['id'].values)
    df2_ids = set(df2['id'].values)
    common_ids = df1_ids.intersection(df2_ids)
    
    print(f"两个文件间重复的id数量: {len(common_ids)}")
    if len(common_ids) > 0 and len(common_ids) <= 20:
        print(f"重复的id: {list(common_ids)}")
    elif len(common_ids) > 20:
        print(f"重复的id (前20个): {list(common_ids)[:20]}")
    
    print(f"df1唯一id数量: {df1['id'].nunique()}")
    print(f"df2唯一id数量: {df2['id'].nunique()}")
    print(f"总唯一id数量: {len(df1_ids.union(df2_ids))}")
    
else:
    print("错误: 其中一个或两个数据框都没有'id'列")
    print(f"df1可用的列: {df1.columns.tolist()}")
    print(f"df2可用的列: {df2.columns.tolist()}")



