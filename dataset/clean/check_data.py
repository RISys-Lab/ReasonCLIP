import pandas as pd
from tqdm import tqdm
import re
import os

llava_cot = "/home/muzammal/Projects/CLIP-R/data/Xkev-LLaVA-CoT-100k/default/train/0000.parquet"
reason_itw = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"
df1 = pd.read_parquet(reason_itw)
df2 = pd.read_parquet(llava_cot)

print("df1 (reason_itw) 信息:")
print(df1.head())
print(df1.columns)
print(f"df1数据形状: {df1.shape}")

print("\ndf2 (llava_cot) 信息:")
print(df2.head())
print(df2.columns)
print(f"df2数据形状: {df2.shape}")

print("\n" + "="*50)
print("根据ID添加image_name列")
print("="*50)

# 检查df2中的id唯一性
df2_unique_ids = df2['id'].nunique()
df2_total_rows = len(df2)
print(f"df2中唯一id数量: {df2_unique_ids} (总行数: {df2_total_rows})")

if df2_unique_ids != df2_total_rows:
    print("⚠️  df2中存在重复的id!")
    duplicates = df2['id'].value_counts()
    duplicates = duplicates[duplicates > 1]
    print(f"重复id数量: {len(duplicates)}")
else:
    print("✅ df2中的id都是唯一的")

# 检查df1中的id在df2中的匹配情况
df1_ids = set(df1['id'])
df2_ids = set(df2['id'])
intersection = df1_ids.intersection(df2_ids)

print(f"\ndf1中的id数量: {len(df1_ids)}")
print(f"df2中的id数量: {len(df2_ids)}")
print(f"共同的id数量: {len(intersection)}")

missing_in_df2 = df1_ids - df2_ids
if len(missing_in_df2) > 0:
    print(f"⚠️  df1中有{len(missing_in_df2)}个id在df2中找不到")
    print(f"缺失的id前5个: {list(missing_in_df2)[:5]}")
else:
    print("✅ df1中的所有id都能在df2中找到")

# 创建id到image的映射
print("\n正在创建image_name列...")
df2_image_mapping = df2.set_index('id')['image'].to_dict()

# 添加image_name列
df1['image_name'] = df1['id'].map(df2_image_mapping)

# 检查结果
null_count = df1['image_name'].isnull().sum()
print(f"成功匹配的行数: {len(df1) - null_count}")
print(f"无法匹配的行数: {null_count}")

print(f"\n添加image_name列后的数据形状: {df1.shape}")
print("新的列结构:")
print(df1.columns.tolist())

print("\n前5行的id和对应的image_name:")
for i in range(5):
    print(f"行{i}:")
    print(f"  id: {df1.iloc[i]['id']}")
    print(f"  image_name: {df1.iloc[i]['image_name']}")
    print()

# 保存结果
output_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"
df1.to_parquet(output_path, index=False)
print(f"✅ 已保存包含image_name列的数据到: {output_path}")
print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")