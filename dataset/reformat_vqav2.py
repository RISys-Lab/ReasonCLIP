from datasets import load_dataset
import os, json
from tqdm import tqdm
from PIL import Image

# ===== 参数设置 =====
dataset_name = "derek-thomas/ScienceQA"
save_dir = "/home/muzammal/Projects/CLIP-R/data/sciqa_images"
save_json = "/home/muzammal/Projects/CLIP-R/data/scienceqa_alpaca.json"
os.makedirs(save_dir, exist_ok=True)

# ===== 加载数据集 =====
ds = load_dataset(dataset_name, split="train")  # 可改成 validation/test

# ===== 功能函数 =====
def idx_to_letter(i):
    return chr(ord("A") + i)

# ===== 转换逻辑 =====
records = []
for i, item in tqdm(enumerate(ds), total=len(ds)):
    image_field = item.get("image", None)
    if image_field is None:
        continue

    image_path = os.path.join(save_dir, f"{i:07d}.jpg")
    try:
        # ScienceQA 中 image 是 PIL.Image 或可转换数组
        if isinstance(image_field, Image.Image):
            image_field.save(image_path)
        else:
            img = Image.fromarray(image_field.convert("RGB"))
            img.save(image_path)
    except Exception as e:
        print(f"保存图片失败 {i}: {e}")
        continue

    # 选项
    choices = item.get("choices", [])
    formatted_choices = [f"{idx_to_letter(j)}. {opt}" for j, opt in enumerate(choices)]

    # 答案索引
    ans_idx = item.get("answer", None)
    if ans_idx is None or ans_idx >= len(choices):
        continue
    answer_text = f"{idx_to_letter(ans_idx)}. {choices[ans_idx]}"

    # instruction
    instruction = (
        "<image>\n"
        + f"Question: {item.get('question', '').strip()}\n"
        + "\n".join(formatted_choices)
    )

    record = {
        "instruction": instruction,
        "input": "",
        "output": answer_text,  # ✅ 直接 "A.xxx"
        "images": [image_path],
    }
    records.append(record)

# ===== 保存 JSON =====
with open(save_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ 已完成转换，共 {len(records)} 条样本")
print(f"图像保存在：{save_dir}/")
print(f"JSON 文件：{save_json}")
