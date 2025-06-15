import os
import ray
from datasets import load_dataset

def create_batch_messages(image_paths, text_prompt):
    """Create batch messages for multiple images with same prompt"""
    messages = []
    for image_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        messages.append(message)
    return messages

SYSTEM_PROMPT_LLAVACOT = """
You are a data generation agent. I need you to generate captions for some images. However, this is not a typical captioning task—there are several special requirements:

1. You will not be provided with the actual image. Instead, you must generate captions solely based on the textual information I give you.
2. The captions you generate should not be simple visual descriptions, but rather reasoning statements about the image content.

I will give you a dialogue between a model and a user. This entire conversation revolves around reasoning about a single image.

Your task is to extract the basic information about the image from this dialogue and summarize the reasoning process. Then, generate several captions that reflect this inferred information.
Tips:
1. You don’t need to perfectly summarize the entire dialogue I provide. Instead, focus on identifying the reasoning present in the conversation, and combine it with the basic content of the image to generate the captions. These captions can differ in their reasoning content and direction.
2. Each caption should reflect higher-level inference based on the conversation, but the level of reasoning should be slightly lower than the full dialogue—more abstract than a literal description, but less complex than the entire reasoning chain.
3. The captions should be short, declarative sentences that directly express the basic image information and the inferred information.

You must generate **three** captions. Keep them concise but meaningful, and **each should be no longer than 80 English words**.
Your output must consist of exactly three captions, with no additional text or output:
1. caption1
2. caption2
3. caption3
"""

USER_PROMPT_LLAVACOT = """
Please provide **three reasoning captions** derived from the conversation that contain **moderate-level reasoning information**.
Please output the three captions following the format in the system prompt.
"""

def ray_prepare_data_llavacot(parquet_dir: str):
    parquet_files = [
        os.path.join(parquet_dir, fname)
        for fname in os.listdir(parquet_dir) 
        if fname.endswith(".parquet")
    ]
    
    # 使用 ray.data.read_parquet 直接读取，避免 arrow_table 兼容性问题
    try:
        ds = ray.data.read_parquet(parquet_files)
    except Exception as e:
        print(f"Direct parquet reading failed: {e}")
        raw_ds = load_dataset(
            "parquet",
            data_files={"train": parquet_files}, 
        )
        # 转换为 pandas DataFrame 再转换为 Ray Dataset
        df = raw_ds['train'].to_pandas()
        ds = ray.data.from_pandas(df)

        # 先过滤数据，更高效
    def should_keep_sample(row):
        filter_keywords = ["chartqa", "geoqa+", "docvqa", "ocr_vqa"]
        image_filename = str(row.get("image", "")).lower()
        
        # 如果文件名包含过滤关键词，则不保留
        for keyword in filter_keywords:
            if keyword in image_filename:
                return False
        return True
    
    print("="*60)
    print(f"Original dataset size: {ds.count()}")
    
    # 先过滤，再转换 - 符合Ray的最佳实践
    ds = ds.filter(should_keep_sample)
    print(f"After filtering: {ds.count()}")
    
    # 然后进行数据转换
    def _extract_fields(row):
        return {
            "id": row["id"],
            "image_path": row["image"],
            "conversations": str(row["conversations"]),
        }
    
    ds = ds.map(_extract_fields)
    print("="*60)
    
    print(ds.schema())  # {'id': str, 'conversations': str}
    return ds

SYSTEM_PROMPT_LLAVACOT_VISUAL = """
You are an image annotation assistant. For each image I provide, you need to generate a concise description. No reasoning is required—just briefly describe the objects and events present in the image.
For each image, generate three captions. They can differ in detail, but must not omit the main subject of the image.
Each caption must be within 70 words.
"""

USER_PROMPT_LLAVACOT_VISUAL = """
Now give me these three captions about the image as the request. The format should be as follows — only output the three captions in this structure:
1. caption1
2. caption2
3. caption3
"""

def ray_prepare_data_llavacot_visual(parquet_dir: str, image_dir: str):
    parquet_files = [
        os.path.join(parquet_dir, fname)
        for fname in os.listdir(parquet_dir) 
        if fname.endswith(".parquet")
    ]
    
    # 使用 ray.data.read_parquet 直接读取，避免 arrow_table 兼容性问题
    try:
        ds = ray.data.read_parquet(parquet_files)
    except Exception as e:
        print(f"Direct parquet reading failed: {e}")
        raw_ds = load_dataset(
            "parquet",
            data_files={"train": parquet_files}, 
        )
        # 转换为 pandas DataFrame 再转换为 Ray Dataset
        df = raw_ds['train'].to_pandas()
        ds = ray.data.from_pandas(df)

        # 先过滤数据，更高效
    def should_keep_sample(row):
        filter_keywords = ["chartqa", "geoqa+", "docvqa", "ocr_vqa"]
        image_filename = str(row.get("image", "")).lower()
        
        # 如果文件名包含过滤关键词，则不保留
        for keyword in filter_keywords:
            if keyword in image_filename:
                return False
        return True
    
    print("="*60)
    print(f"Original dataset size: {ds.count()}")
    
    # 先过滤，再转换 - 符合Ray的最佳实践
    ds = ds.filter(should_keep_sample)
    print(f"After filtering: {ds.count()}")
    
    # 然后进行数据转换
    def _extract_fields(row):
        image_path = os.path.join(image_dir, row["image"])
        # image as path
        return {
            "id": row["id"],
            "image_path": image_path,    
        }
    
    ds = ds.map(_extract_fields)
    print("="*60)
    
    print(ds.schema())  # {'id': str, 'image_path': str}
    return ds
    
def ray_prepare_data_CC12M_visual(data_path: str):
    return None

def ray_prepare_data_parquet_visual(parquet_dir: str):
    parquet_files = [
        os.path.join(parquet_dir, fname)
        for fname in os.listdir(parquet_dir)
        if fname.endswith(".parquet")
    ]
    
    # 使用 ray.data.read_parquet 直接读取，避免 arrow_table 兼容性问题
    try:
        ds = ray.data.read_parquet(parquet_files)
    except Exception as e:
        print(f"Direct parquet reading failed: {e}")
        raw_ds = load_dataset(
            "parquet",
            data_files={"train": parquet_files}, 
        )
        # 转换为 pandas DataFrame 再转换为 Ray Dataset
        df = raw_ds['train'].to_pandas()
        ds = ray.data.from_pandas(df)

    def _extract_fields(row):
        return {
            "data_id": row["data_id"],
            "image_name": row["image_name"],
            "image": row["image"],
            "label": row["label"],
        }

    ds = ds.map(_extract_fields)
    print(ds.schema())  # {'image': binary, 'label': str, 'caption': str}
    return ds


SYSTEM_PROMPT_HAND_VISUAL_ADVICE = """
You are a reasoning agent specialized in egocentric hand-object interaction understanding. Your task is to analyze images captured from a first-person perspective and identify potential issues that may affect 3D hand reconstruction.

Given a single input image, output one concise sentence that describes possible challenges for hand reconstruction, such as occlusions, hand-object interactions, or hand-hand interactions. Be specific about which hand (left/right) is affected and what the issue is. Do not provide explanations or repeat content—output only one short, precise sentence in English.
"""
USER_PROMPT_HAND_VISUAL_ADVICE = """
Please analyze the image and return one sentence describing key challenges for hand reconstruction (e.g., left hand occluded, right hand interacting with an object):
"""


SYSTEM_PROMPT_HAND_VISUAL = """
You are an image understanding agent. Your task is to analyze a first-person perspective image and classify the interaction status of the left and right hands.

Classification rules:
- 0: Only the left hand is interacting with an object
- 1: Only the right hand is interacting with an object
- 2: Both hands are interacting with an object
- 3: Neither hand is interacting with any object

Important notes:
- Occlusion caused by objects must be considered in determining whether a hand is interacting.
- Use your best reasoning based on the visual content to make this decision.

Your response must be a single number: one of [0, 1, 2, -1]. Do not include any explanation or additional text.
"""

USER_PROMPT_HAND_VISUAL = """
Please analyze the first-person perspective image and determine the interaction status of the hands.

Return only the correct label number based on the following:
- 0: Only the left hand is interacting
- 1: Only the right hand is interacting
- 2: Both hands are interacting
- 3: Neither hand is interacting

Do not include any explanation, reasoning, or extra output—just return the number.
"""

def ray_prepare_data_hand_visual(image_dir: str):
    # 获取文件夹中所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for fname in os.listdir(image_dir):
        if any(fname.lower().endswith(ext) for ext in image_extensions):
            image_files.append(fname)
    
    print(f"Found {len(image_files)} image files in {image_dir}")
    
    # 根据图片名筛选，使用提供的正则规则
    filtered_images = []
    for img_name in image_files:
        try:
            # 从图片名中提取id: img_name.split('.')[0].split('_')[-1]
            id_str = img_name.split('.')[0].split('_')[-1]
            id_num = int(id_str)
            
            # 筛选条件：如果 id < 30 或 id > 530，则跳过
            if id_num < 30 or id_num > 530:
                continue
            
            filtered_images.append(img_name)
        except (ValueError, IndexError):
            # 如果无法提取id或转换为整数，跳过此文件
            continue
    
    print("="*60)
    print(f"Original image count: {len(image_files)}")
    print(f"After filtering (id 30-530): {len(filtered_images)}")
    
    # 准备数据列表
    data_list = []
    for img_name in filtered_images:
        image_path = os.path.abspath(os.path.join(image_dir, img_name))
        # id就是图片名（不包含扩展名）
        img_id = img_name.split('.')[0]
        
        data_list.append({
            "id": img_id,
            "image_path": image_path,
        })
    
    # 使用 ray.data.from_items 创建 Ray Dataset
    ds = ray.data.from_items(data_list)
    
    print("="*60)
    print(ds.schema())  # {'id': str, 'image_path': str}
    return ds