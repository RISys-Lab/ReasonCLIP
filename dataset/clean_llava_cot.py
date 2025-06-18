import pandas as pd
import re

# 读取parquet文件
file_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_trp.parquet"
df = pd.read_parquet(file_path)

print(f"文件总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print(f"数据类型: {df.dtypes}")

# 定义函数来提取三个caption
def extract_captions(text):
    """从格式为 '1.xxx 2.xxx 3.xxx' 的文本中提取三个caption"""
    if pd.isna(text):
        return None
    
    # 先用 </think> 分割，取后半段进行处理
    if '</think>' in text:
        parts = text.split('</think>')
        processed_text = parts[-1].strip()  # 取最后一段
    else:
        processed_text = text.strip()
    
    # 方法1: 使用正则表达式匹配 数字.内容 的模式
    pattern1 = r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)'
    matches1 = re.findall(pattern1, processed_text)
    
    # 清理每个caption，去掉前后空格
    captions1 = [match.strip() for match in matches1 if match.strip()]
    
    # 如果方法1成功提取到3个或更多caption，取前3个返回
    if len(captions1) >= 3:
        return captions1[:3]
    
    # 方法2: 处理用\n\n分隔的格式
    # 先按\n\n分割，然后提取每段的数字.内容
    segments = processed_text.split('\n\n')
    captions2 = []
    
    for segment in segments:
        # 匹配每段开头的 数字. 后面的内容
        match = re.match(r'^\d+\.\s*(.+)', segment.strip(), re.DOTALL)
        if match:
            caption = match.group(1).strip()
            # 移除段落内部的换行符，用空格替代
            caption = re.sub(r'\s+', ' ', caption)
            captions2.append(caption)
    
    # 如果方法2成功提取到3个或更多caption，取前3个返回
    if len(captions2) >= 3:
        return captions2[:3]
    
    # 方法3: 处理用换行符分隔的格式（不是\n\n）
    lines = processed_text.split('\n')
    captions3 = []
    
    for line in lines:
        line = line.strip()
        if line and re.match(r'^\d+\.', line):
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                captions3.append(match.group(1).strip())
    
    # 如果方法3成功提取到3个或更多caption，取前3个返回
    if len(captions3) >= 3:
        return captions3[:3]
    
    # 方法4: 处理多组caption的情况，只取第一组的3个
    # 找到所有以数字.开头的位置
    pattern4 = r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]*)*)'
    matches4 = re.findall(pattern4, processed_text, re.MULTILINE)
    
    if len(matches4) >= 6:  # 至少有两组caption
        # 按数字分组
        groups = {}
        for num, content in matches4:
            num = int(num)
            if num not in groups:
                groups[num] = []
            groups[num].append(content.strip())
        
        # 如果有多个"1."，说明有多组，取第一组的1,2,3
        if len(groups.get(1, [])) > 1 and len(groups.get(2, [])) > 1 and len(groups.get(3, [])) > 1:
            captions4 = [
                re.sub(r'\s+', ' ', groups[1][0]),  # 第一个"1."
                re.sub(r'\s+', ' ', groups[2][0]),  # 第一个"2."  
                re.sub(r'\s+', ' ', groups[3][0])   # 第一个"3."
            ]
            return captions4
    
    # 方法5: 处理没有数字编号，只用换行符分隔的三个caption（最后尝试）
    # 按换行符分割，过滤空行，看是否恰好有3个非空段落
    lines5 = [line.strip() for line in processed_text.split('\n') if line.strip()]
    
    if len(lines5) >= 3:
        # 检查每行都不是以数字开头（避免与前面方法重复）
        if not any(re.match(r'^\d+\.', line) for line in lines5):
            captions5 = [re.sub(r'\s+', ' ', line) for line in lines5[:3]]  # 只取前3个
            return captions5
    
    # 所有方法都失败，返回None
    return None

# 定义辅助函数来获取处理后的文本（用于显示）
def get_processed_text(text):
    """获取</think>后面的处理文本"""
    if pd.isna(text):
        return text
    if '</think>' in text:
        parts = text.split('</think>')
        return parts[-1].strip()
    else:
        return text.strip()

# 应用到整个数据框
df['trp'] = df['generated_text'].apply(extract_captions)
df['processed_text'] = df['generated_text'].apply(get_processed_text)

# 统计成功和失败的情况
success_rows = df[df['trp'].notna()]  # 成功提取的行
failed_rows = df[df['trp'].isna()]    # 失败的行（None）

# 查看结果
print("\n" + "="*50)
print("处理后的结果统计:")
print("="*50)
print(f"总行数: {len(df)}")
print(f"成功提取caption的行数: {len(success_rows)}")
print(f"失败(返回None)的行数: {len(failed_rows)}")
print(f"每行caption数量统计:")
caption_counts = df['trp'].apply(lambda x: len(x) if x is not None else 0).value_counts().sort_index()
print(caption_counts)

# 显示失败的例子（完整打印）
if len(failed_rows) > 0:
    print(f"\n前10个失败的例子（完整打印）:")
    print("-" * 80)
    for i, (idx, row) in enumerate(failed_rows.head(10).iterrows()):
        print(f"\n失败例子 {i+1} (行号 {idx}):")
        print(f"</think>后处理文本:")
        print(repr(row['processed_text']))  # 显示处理后的文本
        print(f"处理结果: {row['trp']}")
        print("-" * 80)


output_path = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_trp_cleaned.parquet"
df.to_parquet(output_path)
print(f"\n处理后的数据已保存到: {output_path}")
