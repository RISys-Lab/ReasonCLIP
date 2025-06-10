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

SYSTEM_PROMPT_LLAVACOT = "Give a short description of the image."
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
            "image": row["image"],
            "conversations": row["conversations"],
        }
    
    ds = ds.map(_extract_fields)
    print("="*60)
    
    print(ds.schema())  # {'id': str, 'conversations': str}
    return ds

def ray_prepare_data_CC12M(data_path: str):
    return None

def ray_prepare_data_parquet(parquet_dir: str):
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
        # 调试：检查输入类型
        original_type = type(row["image"]).__name__
        
        if isinstance(row["image"], dict) and "bytes" in row["image"]:
            image_bytes = row["image"]["bytes"]
        elif isinstance(row["image"], bytes):
            image_bytes = row["image"]
        else:
            # 如果是其他格式，先转换为PIL再转为bytes
            if hasattr(row["image"], 'save'):  # PIL Image
                import io as temp_io
                buffer = temp_io.BytesIO()
                row["image"].save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            else:
                # 假设是路径字符串
                with open(row["image"], 'rb') as f:
                    image_bytes = f.read()

        
        # 调试：确认输出类型
        output_type = type(image_bytes).__name__
        if output_type != "bytes":
            print(f"WARNING: Expected bytes, got {output_type} from {original_type}")
        
        # 
        return {
            "image_id": row["image_id"],
            "image": image_bytes,
            "label": row["label"],
            "caption": row["caption"],
        }

    ds = ds.map(_extract_fields)
    print(ds.schema())  # {'image': binary, 'label': str, 'caption': str}
    return ds