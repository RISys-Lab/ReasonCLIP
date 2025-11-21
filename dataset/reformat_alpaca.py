import json

# 读取当前的 alpaca 数据
input_file = "/home/muzammal/Projects/CLIP-R/data/safire_alpaca.json"
output_file = "/home/muzammal/Projects/CLIP-R/data/safire_alpaca.json"

print(f"Loading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print(f"\nCurrent format:")
print(f"  instruction: {data[0]['instruction']}")
print(f"  input: {data[0]['input']}")

# 改回原来的格式
for item in data:
    # 如果 input 是 <image>，说明已经改过了，需要改回去
    if item["input"] == "<image>":
        # 把 <image>\n 加回到 instruction 前面
        item["instruction"] = "<image>\n" + item["instruction"]
        # input 改回空字符串
        item["input"] = ""

print(f"\nRestored format:")
print(f"  instruction: {data[0]['instruction']}")
print(f"  input: {data[0]['input']}")

# 直接覆盖原文件
print(f"\nSaving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Restored {len(data)} samples to original format")

