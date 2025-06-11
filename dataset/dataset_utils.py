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

Your task is to extract the basic information about the image from this dialogue and summarize this long conversation into a few standalone captions.

Each caption should reflect higher-level inference based on the conversation, but the level of reasoning should be slightly lower than the full dialogue—more abstract than a literal description, but less complex than the entire reasoning chain.

The captions should be short, declarative sentences that directly express the inferred information.

You must generate **two** captions. Keep them concise but meaningful, and **each should be no longer than 80 English words**.
"""

USER_PROMPT_LLAVACOT = """
Please provide **two reasoning captions** derived from the conversation that contain **moderate-level reasoning information**.

The format should be as follows — only output the two captions in this structure:

<caption1> the first one <caption1>
<caption2> the second one <caption2>
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
        filter_keywords = ["chartqa",]
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
            "image_path": row["image_path"],
            "conversations": str(row["conversations"]),
        }
    
    ds = ds.map(_extract_fields)
    print("="*60)
    
    print(ds.schema())  # {'id': str, 'conversations': str}
    return ds

SYSTEM_PROMPT_LLAVACOT_VISUAL = """
You are an image annotation assistant. For each image I provide, you need to generate a concise description. No reasoning is required—just briefly describe the objects and events present in the image.
For each image, generate two captions. They can differ in detail, but must not omit the main subject of the image.
Each caption must be within 70 words.
"""

USER_PROMPT_LLAVACOT_VISUAL = """
Now give me these two captions about the image as the request. The format should be as follows — only output the two captions in this structure:

<caption1> the first one <caption1>
<caption2> the second one <caption2>
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
        filter_keywords = ["chartqa",]
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