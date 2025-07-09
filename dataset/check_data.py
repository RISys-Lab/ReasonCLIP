import pandas as pd
from tqdm import tqdm
import re
import os

reason_itw_cls_neg = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/14_000000_000000.parquet"
df1 = pd.read_parquet(reason_itw_cls_neg)
print(df1.head())
print(df1.columns)
print(f"数据形状: {df1.shape}")

# 转换为Excel格式
output_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/14_000000_000000.xlsx"
print(f"正在将数据保存为Excel格式到: {output_path}")

try:
    df1.to_excel(output_path, index=False, engine='openpyxl')
    print(f"成功保存为Excel文件: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"保存Excel文件时出错: {e}")
    print("请确保已安装openpyxl: pip install openpyxl")


