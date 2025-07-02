import pandas as pd
from tqdm import tqdm
import re

reason_pro_file_trp = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_test.parquet"

df_trp = pd.read_parquet(reason_pro_file_trp)
print("df_trp shape:", df_trp.shape)
print("df_trp columns:", df_trp.columns.tolist())
print(df_trp.head())

reason_pro_file_itw = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_best_id.parquet"
df_itw = pd.read_parquet(reason_pro_file_itw)
print("\ndf_itw shape:", df_itw.shape) 
print("df_itw columns:", df_itw.columns.tolist())
print(df_itw.head())

# 根据id匹配，从trp列表中取出对应的元素
print("\n开始处理...")

# 创建新列
df_trp['best_trp'] = None

success_count = 0
error_count = 0

for i in tqdm(range(len(df_trp))):
    try:
        # 获取当前行的id
        current_id = df_trp.iloc[i]['id']
        
        # 在df_itw中找到对应的best_id
        matching_row = df_itw[df_itw['id'] == current_id]
        
        if len(matching_row) > 0:
            best_id = int(matching_row.iloc[0]['best_id'])
            trp_list = df_trp.iloc[i]['trp']
            
            # best_id - 1 作为索引
            index = best_id - 1
            
            # 检查索引是否有效
            if 0 <= index < len(trp_list):
                best_trp = trp_list[index]
                df_trp.at[i, 'best_trp'] = best_trp
                success_count += 1
            else:
                print(f"Index {index} out of range for row {i}, trp_list length: {len(trp_list)}")
                df_trp.at[i, 'best_trp'] = trp_list[0] if len(trp_list) > 0 else ""
                error_count += 1
        else:
            print(f"No matching id found for {current_id} in df_itw")
            df_trp.at[i, 'best_trp'] = df_trp.iloc[i]['trp'][0] if len(df_trp.iloc[i]['trp']) > 0 else ""
            error_count += 1
            
    except Exception as e:
        print(f"Error processing row {i}: {str(e)}")
        df_trp.at[i, 'best_trp'] = df_trp.iloc[i]['trp'][0] if len(df_trp.iloc[i]['trp']) > 0 else ""
        error_count += 1

# 打印统计结果
print(f"\n处理完成!")
print(f"总行数: {len(df_trp)}")
print(f"成功处理: {success_count}")
print(f"错误/异常: {error_count}")
print(f"成功率: {success_count/len(df_trp)*100:.2f}%")

# 保存结果
output_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/llavacot_test_with_best_trp.parquet"
df_trp.to_parquet(output_file)
print(f"结果已保存到: {output_file}")

# 显示几个样例结果
print("\n前5个样例:")
for i in range(min(5, len(df_trp))):
    print(f"Row {i}:")
    print(df_trp.iloc[i])

