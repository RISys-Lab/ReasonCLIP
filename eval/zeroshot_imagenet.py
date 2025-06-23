import torch
import open_clip
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from torch.utils.data import DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading


class ImageNetDataset(torch.utils.data.Dataset):
    """优化的ImageNet数据集类，支持快速批处理"""
    def __init__(self, ds, processor, device, max_samples=None, preload_images=False, num_workers=4):
        self.ds = ds
        self.processor = processor
        self.device = device
        self.max_samples = max_samples
        self.data_list = []
        self.preload_images = preload_images
        self.images_cache = {}
        
        print("Loading dataset...")
        if preload_images:
            print("🚀 Preloading images to memory for maximum speed...")
        
        # 决定是否使用多线程加载
        use_threading = max_samples and max_samples <= 10000  # 小于10K时使用多线程
        
        if use_threading:
            self._load_with_threading(num_workers)
        else:
            self._load_sequential()
    
    def _load_with_threading(self, num_workers):
        """使用多线程加载数据"""
        print(f"Using {num_workers} threads for faster data loading...")
        
        # 首先收集所有样本
        samples_list = []
        for i, sample in enumerate(tqdm(self.ds, desc="Collecting samples")):
            if self.max_samples and i >= self.max_samples:
                break
            samples_list.append((i, sample))
        
        def process_sample(item):
            i, sample = item
            try:
                # 预处理图片（如果启用）
                if self.preload_images:
                    image_data = sample["jpg"]
                    img = image_data.convert("RGB") if hasattr(image_data, "convert") else Image.open(io.BytesIO(image_data)).convert("RGB")
                    self.images_cache[i] = img
                return sample
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                return None
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_sample, samples_list),
                total=len(samples_list),
                desc="🔥 Processing with threads"
            ))
            self.data_list = [r for r in results if r is not None]
        
        print(f"✅ Loaded {len(self.data_list)} samples with threading")
    
    def _load_sequential(self):
        """顺序加载数据（大数据集）"""
        print("Using sequential loading for large dataset...")
        for i, sample in enumerate(tqdm(self.ds, desc="Loading samples")):
            if self.max_samples and i >= self.max_samples:
                break
            self.data_list.append(sample)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # 使用预加载的图片或实时解码
        if self.preload_images and idx in self.images_cache:
            img = self.images_cache[idx]
        else:
            image_data = sample["jpg"]
            img = image_data.convert("RGB") if hasattr(image_data, "convert") else Image.open(io.BytesIO(image_data)).convert("RGB")
        
        label = int(sample["cls"])
        return img, label

def collate_fn(batch, processor):
    """批处理整理函数 - 保持在CPU，避免CUDA多进程问题"""
    images, labels = zip(*batch)
    # 批量处理图像，保持在CPU
    image_inputs = processor(images=list(images), return_tensors="pt")
    labels = torch.tensor(labels)
    return image_inputs, labels

def run_zero_shot_imagenet_wds_optimized(
    model_id="safeclip_vit-l_14", 
    split="validation", 
    max_samples=None,
    batch_size=32,  # 批处理大小
    num_workers=4,  # 数据加载进程数
    use_amp=True,   # 混合精度
    pin_memory=True, # 内存固定
    fast_mode=False, # 快速模式
    device=None      # 设备配置
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_id}")
    print(f"Using batch_size: {batch_size}, num_workers: {num_workers}, AMP: {use_amp}")
    
    # 快速模式优化
    if fast_mode or (max_samples and max_samples <= 5000):
        print("🚀 FAST MODE ENABLED")
        preload_images = max_samples and max_samples <= 2000  # 小数据集预加载图片
        batch_size = min(batch_size * 2, 128)  # 增大batch size
        num_workers = min(num_workers * 2, 16) # 增加worker数量
        data_loading_threads = 8  # 数据加载线程数
        print(f"Fast mode: batch_size={batch_size}, num_workers={num_workers}, preload={preload_images}")
    else:
        preload_images = False
        data_loading_threads = 4

    # 模型初始化
    if model_id == "safeclip_vit-l_14_336":
        print("Loading safeclip_vit-l_14_336 model")
        model = CLIPModel.from_pretrained(f'aimagelab/{model_id}')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif model_id == "safeclip_vit-h_14":
        model = CLIPModel.from_pretrained(f'aimagelab/{model_id}')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)

    model.to(device).eval()
    
    # 模型编译优化 (PyTorch 2.0+)
    if hasattr(torch, 'compile') and fast_mode:
        print("Compiling model for faster inference...")
        try:
            model = torch.compile(model)
        except:
            print("Model compilation failed, continuing without it")
    
    # 启用混合精度
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 加载类名 & 构建文本特征（一次性计算）
    classnames = [line.strip() for line in open("/home/muzammal/Projects/safe_proj/safe-clip/imagenet/imagenet_classes.txt")]
    prompts = [f"a photo of a {name}" for name in classnames]

    print("Computing text features...")
    with torch.no_grad():
        if 'clip' in model_id:
            inputs = processor(text=prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    text_features = model.get_text_features(inputs.input_ids.to(device))
            else:
                text_features = model.get_text_features(inputs.input_ids.to(device))
        else:
            pass
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 加载数据集（使用优化的数据集类）
    ds = load_dataset("timm/imagenet-1k-wds", split=split, streaming=True)
    dataset = ImageNetDataset(
        ds, processor, device, max_samples, 
        preload_images=preload_images,
        num_workers=data_loading_threads
    )
    
    # 创建优化的DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # 保持worker进程活跃
        prefetch_factor=2 if num_workers > 0 else None,  # 预取因子
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    top1, top5, total = 0, 0, 0

    print("Starting evaluation...")
    with torch.no_grad():
        for image_inputs, labels in tqdm(dataloader, desc="Evaluating"):
            # 将数据移动到GPU (在主进程中进行)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            labels = labels.to(device)
            batch_size_current = labels.size(0)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    image_features = model.get_image_features(**image_inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = image_features @ text_features.T
            else:
                image_features = model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
            
            # 批量计算top-k
            topk = logits.topk(5, dim=-1).indices  # [batch_size, 5]
            
            # 批量统计准确率
            top1_batch = (topk[:, 0] == labels).sum().item()
            top5_batch = (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            top1 += top1_batch
            top5 += top5_batch
            total += batch_size_current

    print(f"Zero-Shot {model_id} on ImageNet-WDS ({split}):")
    print(f"Top-1 (R@1): {top1 / total * 100:.2f}%")
    print(f"Top-5 (R@5): {top5 / total * 100:.2f}%")
    print(f"Total samples: {total}")
    
    # 保存结果
    model_name = model_id.replace("/", "_")
    with open(f"results_{model_name}_optimized.txt", "w") as f:
        f.write(f"Zero-Shot {model_id} on ImageNet-WDS ({split}):\n")
        f.write(f"Top-1 (R@1): {top1 / total * 100:.2f}%\n")
        f.write(f"Top-5 (R@5): {top5 / total * 100:.2f}%\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Batch size: {batch_size}, AMP: {use_amp}\n")
        f.write(f"Fast mode: {fast_mode}, Preload: {preload_images}\n")

    return {
        'top1_accuracy': top1 / total * 100,
        'top5_accuracy': top5 / total * 100,
        'total_samples': total
    }

def run_zero_shot_imagenet_wds(model_id="safeclip_vit-l_14", split="validation", max_samples=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_id}")

    # 模型初始化


    if model_id == "safeclip_vit-l_14_336":
        print("Loading safeclip_vit-l_14_336 model")
        model = CLIPModel.from_pretrained(f'aimagelab/{model_id}')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    elif model_id == "safeclip_vit-h_14":
        model = CLIPModel.from_pretrained(f'aimagelab/{model_id}')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)

    model.to(device).eval()

    # 加载类名 & 构建文本特征
    classnames = [line.strip() for line in open("/home/muzammal/Projects/safe_proj/safe-clip/imagenet/imagenet_classes.txt")]
    prompts = [f"a photo of a {name}" for name in classnames]

    with torch.no_grad():
        if 'clip' in model_id:
            inputs = processor(text=prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
            text_features = model.get_text_features(inputs.input_ids.to(device))
        else:
            pass
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 加载 WDS 数据集（自动流式）
    ds = load_dataset("timm/imagenet-1k-wds", split=split, streaming=True)

    top1, top5, total = 0, 0, 0

    for sample in tqdm(ds, desc="Evaluating", total=max_samples or 50000):
        if max_samples and total >= max_samples:
            break
        
        image_data = sample["jpg"]
        img = image_data.convert("RGB") if hasattr(image_data, "convert") else Image.open(io.BytesIO(image_data)).convert("RGB")
        
        image_tensor = processor(images=img, return_tensors="pt").to(device)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**image_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            topk = logits.topk(5, dim=-1).indices.squeeze(0)

            label = int(sample["cls"])
            top1 += (topk[0].item() == label)
            top5 += (label in topk)
            total += 1

    print(f"Zero-Shot {model_id} on ImageNet-WDS ({split}):")
    print(f"Top-1 (R@1): {top1 / total * 100:.2f}%")
    print(f"Top-5 (R@5): {top5 / total * 100:.2f}%")
    with open(f"results_{model_id}.txt", "w") as f:
        f.write(f"Zero-Shot {model_id} on ImageNet-WDS ({split}):\n")
        f.write(f"Top-1 (R@1): {top1 / total * 100:.2f}%\n")
        f.write(f"Top-5 (R@5): {top5 / total * 100:.2f}%\n")

if __name__ == "__main__":
    # 使用优化版本 - 快速模式
    print("🚀 Running optimized zero-shot evaluation with fast mode...")
    result = run_zero_shot_imagenet_wds_optimized(
        model_id="openai/clip-vit-large-patch14", 
        split="validation",
        batch_size=64,   # 更大的batch size
        num_workers=16,  # 更多worker进程
        use_amp=True,    # 启用混合精度
        fast_mode=True,  # 🔥 启用快速模式
        max_samples=None, # 评估完整数据集，或设置为1000来快速测试
        device="cuda:3"  # 指定GPU设备
    )
    
    print(f"\n🎯 Final Results:")
    print(f"Top-1: {result['top1_accuracy']:.2f}%")
    print(f"Top-5: {result['top5_accuracy']:.2f}%")
    print(f"Samples: {result['total_samples']}")
    
    # 原版本（保留兼容性）
    # run_zero_shot_imagenet_wds(model_id="fesvhtr/clip_r_best_model_demo_0621_192211", split="validation")
