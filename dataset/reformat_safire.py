from datasets import load_dataset
import random
import os
from PIL import Image
import io
from tqdm import tqdm
raw_ds = load_dataset("fesvhtr/iferniu", split="train")

# 创建图片保存目录
image_dir = "/home/muzammal/Projects/CLIP-R/data/safire_image"
os.makedirs(image_dir, exist_ok=True)

instruction_list = [
    "Describe the image briefly, paying special attention to the fire and smoke.",
    "Give an overall description of the image, especially focusing on any fire or smoke.",
    "Provide a short summary of what is shown in the image, with attention to fire and smoke.",
    "Describe what you see in the image, noting details about the fire and smoke.",
    "Offer a general description of the scene, particularly the fire and smoke conditions.",
    "Summarize the image, mentioning both the general environment and any visible fire or smoke.",
    "Describe the scene in the image, focusing on the appearance of fire and smoke if present.",
    "Provide a concise description of the image, highlighting flames and smoke.",
    "Give an overall description of what is happening in the picture, emphasizing the fire and smoke.",
    "Describe the contents of the image, especially features related to fire and smoke.",
    "Summarize the visual scene, with special attention to the fire intensity and smoke behavior.",
    "Describe the image as a whole, and mention any fire or smoke that can be observed.",
    "Provide a brief description of the scene, including observations about fire and smoke.",
    "Give a description of the image, focusing on how the fire and smoke appear.",
    "Describe the image generally, but include details about the fire and smoke areas.",
    "Summarize the picture in a few words, noting both the general scene and the fire or smoke.",
    "Describe what the image shows overall, with extra focus on fire and smoke elements.",
    "Provide a short explanation of the image content, particularly related to fire and smoke.",
    "Give a description of the scene, highlighting both the environment and any visible fire or smoke.",
    "Describe the overall image, paying close attention to the fire and smoke characteristics."
]

print(f"Total samples: {len(raw_ds)}")

alpaca_data = []
llava_data = []

for idx, row in tqdm(enumerate(raw_ds), total=len(raw_ds)):
    image_filename = f"{idx:07d}.jpg"
    image_path_full = os.path.join(image_dir, image_filename)
    image_path_relative = f"safire_image/{image_filename}"
    
    # 从 row['image'] 中提取图片字节数据
    image_data = row['image']
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # 标准 HF 格式: {'bytes': b'...', 'path': '...'}
        img_bytes = image_data['bytes']
    else:
        # 如果直接是 bytes
        img_bytes = image_data
    
    # 将字节数据转换为 PIL Image 并保存为 JPG
    image = Image.open(io.BytesIO(img_bytes))
    image = image.convert('RGB')  # 确保是 RGB 格式
    image.save(image_path_full, 'JPEG', quality=95)
    
    # 随机选择一个instruction
    selected_instruction = random.choice(instruction_list)
    
    # ========== 格式1: Alpaca 格式 ==========
    alpaca_row = {
        "instruction": selected_instruction,
        "input": "<image>",
        "output": row["caption"],
        "images": [image_path_relative]
    }
    alpaca_data.append(alpaca_row)
    
    # ========== 格式2: LLaVA 格式 ==========
    llava_row = {
        "id": f"safire_{idx:05d}",
        "image": image_path_relative,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + selected_instruction
            },
            {
                "from": "gpt",
                "value": row["caption"]
            }
        ]
    }
    llava_data.append(llava_row)

print(f"\n✅ Completed! Saved {len(alpaca_data)} images to {image_dir}")

# 保存 Alpaca 格式的数据为 JSON
import json
alpaca_output = "/home/muzammal/Projects/CLIP-R/data/safire_alpaca.json"
with open(alpaca_output, 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
print(f"✅ Saved Alpaca format to {alpaca_output}")
print(f"   Sample: {alpaca_data[0]}")

# 保存 LLaVA 格式的数据为 JSON
# llava_output = "/home/muzammal/Projects/CLIP-R/data/safire_llava.json"
# with open(llava_output, 'w', encoding='utf-8') as f:
#     json.dump(llava_data, f, ensure_ascii=False, indent=2)
# print(f"✅ Saved LLaVA format to {llava_output}")
# print(f"   Sample: {llava_data[0]}")