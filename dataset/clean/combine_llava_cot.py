import pandas as pd
import os

# 文件路径
tb_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_tb_cleaned.parquet"
trp_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_trp_cleaned.parquet"

# 检查文件是否存在
if not os.path.exists(tb_file):
    print(f"错误: 找不到文件 {tb_file}")
    exit(1)

if not os.path.exists(trp_file):
    print(f"错误: 找不到文件 {trp_file}")
    exit(1)

# 读取两个文件
print("读取tb文件...")
df_tb = pd.read_parquet(tb_file)
print(f"tb文件行数: {len(df_tb)}")
print(f"tb文件列名: {list(df_tb.columns)}")

print("\n读取trp文件...")
df_trp = pd.read_parquet(trp_file)
print(f"trp文件行数: {len(df_trp)}")
print(f"trp文件列名: {list(df_trp.columns)}")

# 检查是否有id列
if 'id' not in df_tb.columns:
    print("错误: tb文件中没有id列")
    exit(1)

if 'id' not in df_trp.columns:
    print("错误: trp文件中没有id列")
    exit(1)

# 给generated_text加前缀
print("\n添加前缀...")
df_tb['generated_text_tb'] = 'tb: ' + df_tb['generated_text'].astype(str)
df_trp['generated_text_trp'] = 'trp: ' + df_trp['generated_text'].astype(str)

# 准备合并的列
tb_columns = ['id', 'image_path', 'tb', 'generated_text_tb']
trp_columns = ['id', 'trp', 'generated_text_trp']

# 选择需要的列
df_tb_merge = df_tb[tb_columns].copy()
df_trp_merge = df_trp[trp_columns].copy()

# 按id合并
print("\n按id合并数据...")
df_combined = pd.merge(df_tb_merge, df_trp_merge, on='id', how='inner')

print(f"合并后行数: {len(df_combined)}")

# 过滤掉trp为None的行
print("\n过滤trp为None的行...")
original_count = len(df_combined)
df_combined = df_combined[df_combined['trp'].notna()]
filtered_count = len(df_combined)

print(f"过滤前行数: {original_count}")
print(f"过滤后行数: {filtered_count}")
print(f"舍弃的行数: {original_count - filtered_count}")
print(f"合并后列名: {list(df_combined.columns)}")

# 检查合并结果
print(f"\ntb成功提取的行数: {df_combined['tb'].notna().sum()}")
print(f"trp成功提取的行数: {df_combined['trp'].notna().sum()}")  # 现在应该都是成功的
print(f"两者都成功的行数: {(df_combined['tb'].notna() & df_combined['trp'].notna()).sum()}")

# 显示前几行示例
print("\n前3行示例:")
for i in range(min(3, len(df_combined))):
    print(f"\n第{i+1}行:")
    print(f"id: {df_combined.iloc[i]['id']}")
    print(f"image_path: {df_combined.iloc[i]['image_path']}")
    print(f"tb: {df_combined.iloc[i]['tb']}")
    print(f"trp: {df_combined.iloc[i]['trp']}")
    print(f"generated_text_tb: {df_combined.iloc[i]['generated_text_tb'][:100]}...")
    print(f"generated_text_trp: {df_combined.iloc[i]['generated_text_trp'][:100]}...")
    print("-" * 50)


output_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet"
df_combined.to_parquet(output_path, index=False)
print(f"\n合并后的数据已保存到: {output_path}")

# 保存统计信息
stats = {
    'total_rows': len(df_combined),
    'tb_success': df_combined['tb'].notna().sum(),
    'trp_success': df_combined['trp'].notna().sum(),  # 现在应该等于total_rows
    'both_success': (df_combined['tb'].notna() & df_combined['trp'].notna()).sum(),
    'tb_failed': df_combined['tb'].isna().sum()  # 只统计tb失败的数量
}

print(f"\n详细统计:")
print(f"总行数: {stats['total_rows']}")
print(f"tb成功: {stats['tb_success']} ({stats['tb_success']/stats['total_rows']*100:.1f}%)")
print(f"trp成功: {stats['trp_success']} ({stats['trp_success']/stats['total_rows']*100:.1f}%)")  # 应该是100%
print(f"两者都成功: {stats['both_success']} ({stats['both_success']/stats['total_rows']*100:.1f}%)")
print(f"tb失败但trp成功: {stats['tb_failed']} ({stats['tb_failed']/stats['total_rows']*100:.1f}%)")
