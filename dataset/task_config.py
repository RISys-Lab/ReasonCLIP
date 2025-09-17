import os
import ray
from datasets import load_dataset
from dataset.prompts import *
import re

class LlavaCotTask:
    def __init__(self, temperature, max_tokens, top_p, top_k):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.SYSTEM_PROMPT = SYSTEM_PROMPT_LLAVACOT
        self.USER_PROMPT = USER_PROMPT_LLAVACOT

    def prepare_dataset(self, parquet_dir, image_dir):
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

    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT
        user_prompt = self.USER_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": user_prompt + "\n" + row["conversations"]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
            },
        }
    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "generated_text": row["generated_text"],
        }

class HandVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.SYSTEM_PROMPT = SYSTEM_PROMPT_HAND_VISUAL_ADVICE_V2
        self.USER_PROMPT = USER_PROMPT_HAND_VISUAL_ADVICE_V2
    def prepare_dataset(self, parquet_dir, image_dir):
        # 获取图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        # 支持单个目录或目录列表
        if isinstance(image_dir, list):
            image_dirs = image_dir
        else:
            image_dirs = [image_dir]
        
        # 对每个目录执行图片查找逻辑
        for current_dir in image_dirs:
            print(f"Processing directory: {current_dir}")
            
            # 首先检查根目录是否有图片文件
            root_files = os.listdir(current_dir)
            root_images = [f for f in root_files if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            if root_images:
                # 如果根目录有图片，只使用根目录的图片
                print(f"  Found images in root directory, using only root level images")
                for fname in root_images:
                    abs_path = os.path.abspath(os.path.join(current_dir, fname))
                    image_files.append((fname, fname, abs_path))
            else:
                # 如果根目录没有图片，递归遍历所有子文件夹
                print(f"  No images in root directory, searching subdirectories recursively")
                for root, dirs, files in os.walk(current_dir):
                    for fname in files:
                        if any(fname.lower().endswith(ext) for ext in image_extensions):
                            # 存储相对于current_dir的相对路径和完整的绝对路径
                            rel_path = os.path.relpath(os.path.join(root, fname), current_dir)
                            abs_path = os.path.abspath(os.path.join(root, fname))
                            image_files.append((fname, rel_path, abs_path))
            
            print(f"  Found {len([f for f in image_files if f[2].startswith(os.path.abspath(current_dir))])} images in {current_dir}")
        
        print(f"Found {len(image_files)} image files total from all directories")
        
        # 根据图片名筛选，使用提供的正则规则
        filtered_images = []
        # for img_tuple in image_files:
        #     try:
        #         # img_tuple 是 (fname, rel_path, abs_path)
        #         fname, rel_path, abs_path = img_tuple
        #         # 从图片名中提取id: fname.split('.')[0].split('_')[-1]
        #         id_str = fname.split('.')[0].split('_')[-1]
        #         id_num = int(id_str)
                
        #         # 筛选条件：如果 id < 30 或 id > 530，则跳过
        #         if id_num < 30 or id_num > 530:
        #             continue
                
        #         filtered_images.append(img_tuple)
        #     except (ValueError, IndexError):
        #         # 如果无法提取id或转换为整数，跳过此文件
        #         continue
        
        print("="*60)
        print(f"Original image count: {len(image_files)}")
        print(f"After filtering (id 30-530): {len(filtered_images)}")
        
        # 暂时使用所有图片（注释掉筛选逻辑）
        filtered_images = image_files
        
        # 准备数据列表
        data_list = []
        for img_name, rel_path, abs_path in filtered_images:
            # id就是图片名（不包含扩展名）
            img_id = img_name.split('.')[0]
            
            data_list.append({
                "id": img_id,
                "image_path": abs_path,  # 使用绝对路径
            })
        
        # 使用 ray.data.from_items 创建 Ray Dataset
        ds = ray.data.from_items(data_list)
        
        print("="*60)
        print(ds.schema())  # {'id': str, 'image_path': str}
        return ds


    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT
        user_prompt = self.USER_PROMPT
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "generated_text": row["generated_text"],
        }
    
class LlavaCotVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.SYSTEM_PROMPT = SYSTEM_PROMPT_LLAVACOT_VISUAL
        self.USER_PROMPT = USER_PROMPT_LLAVACOT_VISUAL
    def prepare_dataset(self, parquet_dir, image_dir):
        image_dir = image_dir[0]
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

    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT
        user_prompt = self.USER_PROMPT
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "generated_text": row["generated_text"],
        }

class CC12MVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.SYSTEM_PROMPT_CC12M_tb_visual = """
        You are an image annotation assistant. For each image I provide, you need to generate some concise descriptions. No reasoning is required—just briefly describe the objects and events present in the image.
        For each image, I will provide a raw and draft caption (with very limited words), it will provides some additional information to help you.
        For each image, generate three captions. They can differ in detail, but must not omit the main subject of the image.
        Each caption must be short and concise within 50 words.
        """

        self.USER_PROMPT_CC12M_tb_visual = """
        Now give me these three captions about the image as the request. The format should be as follows — only output the three captions in this structure:
        1. caption1
        2. caption2
        3. caption3
        """


        self.SYSTEM_PROMPT_CC12M_trp_visual = """
        """

        self.USER_PROMPT_CC12M_trp_visual = """
        """

        self.SYSTEM_PROMPT_CC12M_cls_visual = """
        """

        self.USER_PROMPT_CC12M_cls_visual = """
        """


    def prepare_dataset(self, parquet_dir, image_dir):
        image_dir = image_dir[0]
        # 直接读取parquet文件, each 2 million rows
        parquet_files = parquet_dir
        
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
        
        print("="*60)
        print(f"Dataset size: {ds.count()}")
        
        # 然后进行数据转换
        def _extract_fields(row):
            image_name = row["id"] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            # image as path
            return {
                "id": row["id"],
                "image_path": image_path,
                "raw_caption": row["raw_caption"],
            }
        
        ds = ds.map(_extract_fields)
        print("="*60)
        
        print(ds.schema())  # {'id': str, 'image_path': str, 'raw_caption': str}
        return ds

    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT_CC12M_tb_visual
        user_prompt = self.USER_PROMPT_CC12M_tb_visual
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the raw and draft caption for the image: " + row["raw_caption"] + "\n" + user_prompt},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "generated_text": row["generated_text"],
            "raw_caption": row["raw_caption"],
        }

class CC12MtbVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.SYSTEM_PROMPT_CC12M_tb_visual = """
        You are an image annotation assistant. For each image I provide, you need to generate some concise descriptions. No reasoning is required—just briefly describe the objects and events present in the image.
        For each image, I will provide a raw and draft caption (with very limited words), it will provides some additional information to help you.
        For each image, generate three captions. They can differ in detail, but must not omit the main subject of the image.
        Each caption must be short and concise within 50 words.
        """

        self.USER_PROMPT_CC12M_tb_visual = """
        Now give me these three captions about the image as the request. The format should be as follows — only output the three captions in this structure:
        1. caption1
        2. caption2
        3. caption3
        """


    def prepare_dataset(self, parquet_dir, image_dir):
        image_dir = image_dir[0]
        # 直接读取parquet文件, each 2 million rows
        parquet_files = parquet_dir
        
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
        
        print("="*60)
        print(f"Dataset size: {ds.count()}")
        
        # 然后进行数据转换
        def _extract_fields(row):
            image_name = row["id"] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            # image as path
            return {
                "id": row["id"],
                "image_path": image_path,
                "raw_caption": row["raw_caption"],
            }
        
        ds = ds.map(_extract_fields)
        print("="*60)
        
        print(ds.schema())  # {'id': str, 'image_path': str, 'raw_caption': str}
        return ds

    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT_CC12M_tb_visual
        user_prompt = self.USER_PROMPT_CC12M_tb_visual
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the raw and draft caption for the image: " + row["raw_caption"] + "\n" + user_prompt},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "generated_text": row["generated_text"],
            "raw_caption": row["raw_caption"],
        }

class CC12MtrlVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.SYSTEM_PROMPT_CC12M_trl_visual = """
        You are an image annotation assistant. Each time I will provide you with one image and a simple description of the image. Here is your task:
        1. You need to generate three common level reasoning captions about the image, the reasoning level should be higher than the simple description.
        2. Each time, you have the ability and right to choose the fitting direction of the reasoning based on the image content. The simple description is just for your reference.
        3. It's not that your captions make people to think, the level of reasoning in your captions is the result of simple human thinking.
        4. If the content permits, the reasoning approaches for the three captions should be somewhat distinct, though all should remain at a basic level of reasoning without requiring overly complex chains of inference.
        5. Finally, you need to generate three captions, each caption should be short and concise within 50 words, no more than 2 short sentences.
        """

        self.USER_PROMPT_CC12M_trl_visual = """
        Now give me the three reasoning captions about the image as the request, each caption should be short and concise within 50 words, no more than 2 short sentences.
        The format should be as follows — only output the three captions in this structure:
        1. caption1
        2. caption2
        3. caption3
        """


    def prepare_dataset(self, parquet_dir, image_dir):
        # 直接读取parquet文件, each 2 million rows
        parquet_files = parquet_dir
        
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
        
        print("="*60)
        print(f"Dataset size: {ds.count()}")
        
        # 然后进行数据转换
        def _extract_fields(row):
            # image as path
            s = (row["tb"] if isinstance(row["tb"], str) else str(row["tb"])).replace("\\n", "\n")

            m = re.search(r'^\s*1\.\s*(.*?)(?=\n\s*\d+\.|\Z)', s, flags=re.S | re.M)
            if m:
                tb1 = m.group(1).strip()
            else:
                m = re.search(r'^\s*1\.\s*(.*)$', s, flags=re.M)
                tb1 = (m.group(1).strip() if m else s.strip())
            return {
                "id": row["id"],
                "image_path": row["image_path"],
                "tb": tb1,
            }
        
        ds = ds.map(_extract_fields)
        print("="*60)
        
        print(ds.schema())  # {'id': str, 'image_path': str, 'tb': list}
        return ds

    def preprocess(self, row):
        system_prompt = self.SYSTEM_PROMPT_CC12M_trl_visual
        user_prompt = self.USER_PROMPT_CC12M_trl_visual
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the simple description for the image: " + row["tb"] + "\n" + user_prompt},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "tb": row["tb"],
            "generated_text": row["generated_text"],
        }

class ReasonItwClsVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.SYSTEM_PROMPT = SYSTEM_PROMPT_REASON_ITW_CLS
        self.USER_PROMPT = USER_PROMPT_REASON_ITW_CLS


    def prepare_dataset(self, parquet_dir, image_dir):
        parquet_files = parquet_dir
        
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
        
        print("="*60)
        print(f"Dataset size: {ds.count()}")
        
        # 然后进行数据转换
        def _extract_fields(row):
            # image as path
            return {
                "id": row["id"],
                "image_path": row["image_path"],
                "trp": row["trp"],
            }
        
        ds = ds.map(_extract_fields)
        print("="*60)
        
        print(ds.schema())  # {'id': str, 'image_path': str, 'trp': str}
        return ds

    def preprocess(self, row):
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        trp = row["trp"]
        trps = ""
        for i in range(len(trp)):
            trps += f"{i+1}: {trp[i]}\n"
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.USER_PROMPT + "\n" + trps},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "trp": row["trp"],
            "generated_text": row["generated_text"],
        }
    
class ReasonItwClsNegVisualTask:
    
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.SYSTEM_PROMPT = SYSTEM_PROMPT_REASON_ITW_CLS_NEG
        self.USER_PROMPT = USER_PROMPT_REASON_ITW_CLS_NEG


    def prepare_dataset(self, parquet_dir, image_dir):
        parquet_files = parquet_dir
        
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
        
        print("="*60)
        print(f"Dataset size: {ds.count()}")
        
        # 然后进行数据转换
        def _extract_fields(row):
            # image as path
            return {
                "id": row["id"],
                "image_path": row["image_path"],
                "best_trp": row['best_trp'],
            }
        
        ds = ds.map(_extract_fields)
        print("="*60)
        
        print(ds.schema())
        return ds

    def preprocess(self, row):
        from PIL import Image
        image = Image.open(row["image_path"])
        image = image.convert('RGB')
        best_trp = row["best_trp"]
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.USER_PROMPT + "\n" + best_trp},
                    {"type": "image", "image": image}
                ]
            },
        ]
        return {
            "messages": messages,
            "sampling_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        }

    def postprocess(self, row):
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "best_trp": row["best_trp"],
            "generated_text": row["generated_text"],
        }

# 添加 Cyber1Task 类的定义（从代码中看起来缺失了）
class Cyber1Task:
    def __init__(self, temperature, max_tokens, top_p):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        # 这里需要添加实际的实现，暂时留空
        pass

    def prepare_dataset(self, parquet_dir, image_dir):
        # 需要根据实际需求实现
        pass

    def preprocess(self, row):
        # 需要根据实际需求实现
        pass

    def postprocess(self, row):
        # 需要根据实际需求实现
        pass

# 任务注册表 - 将任务名称映射到对应的类
TASK_REGISTRY = {
    "llavacot": LlavaCotTask,
    "cyber1": Cyber1Task,
    "llavacot_visual": LlavaCotVisualTask,
    "hand_visual": HandVisualTask,
    "cc12m_tb_visual": CC12MtbVisualTask,
    "reason_itw_cls_visual": ReasonItwClsVisualTask,
    "reason_itw_cls_neg_visual": ReasonItwClsNegVisualTask,
    "cc12m_trl_visual": CC12MtrlVisualTask,
}

def create_task_config(task_name, temperature, max_tokens, top_p, top_k=None):
    """
    根据任务名称创建对应的任务配置对象
    
    Args:
        task_name: 任务名称
        temperature: 采样温度
        max_tokens: 最大生成tokens
        top_p: top-p采样参数
        top_k: top-k采样参数（可选）
    
    Returns:
        对应的任务配置对象
    
    Raises:
        ValueError: 当任务名称不存在时
    """
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Invalid task: {task_name}. Available tasks: {available_tasks}")
    
    task_class = TASK_REGISTRY[task_name]
    
    # 检查构造函数需要的参数
    import inspect
    sig = inspect.signature(task_class.__init__)
    params = list(sig.parameters.keys())[1:]  # 排除 self
    
    # 根据参数构造kwargs
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    # 只有当构造函数需要top_k参数时才添加
    if "top_k" in params and top_k is not None:
        kwargs["top_k"] = top_k
    
    return task_class(**kwargs)
