import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader
import json
import os
import requests


class RetrievalDataset(torch.utils.data.Dataset):
    """Standard retrieval dataset for COCO/Flickr30K with Karpathy 5-caption support"""
    def __init__(self, dataset, processor, split="test", max_samples=None, use_all_captions=True, local_image_dir=None):
        self.processor = processor
        self.use_all_captions = use_all_captions
        self.local_image_dir = local_image_dir  # 本地图片目录
        self.image_data = []  # Store image info
        self.caption_data = []  # Store all captions
        self.image_to_captions = {}  # Map image index to caption indices
        self.caption_to_image = {}  # Map caption index to image index
        
        # Set text max length based on processor type
        proc_name = processor.__class__.__name__.lower()
        if "siglip" in proc_name:
            self.text_max_len = 64  # SigLIP uses 64
        else:
            self.text_max_len = 77  # CLIP uses 77
        
        print(f"Loading {split} split...")
        print(f"Karpathy evaluation mode: {'5 captions per image' if use_all_captions else '1 caption per image'}")
        if local_image_dir:
            print(f"📁 使用本地图片目录: {local_image_dir}")
        
        caption_idx = 0
        for img_idx, sample in enumerate(tqdm(dataset)):
            if max_samples and img_idx >= max_samples:
                break
            
            # 调试：打印第一个样本的字段
            # if img_idx == 0:
            #     print(f"📋 样本字段: {list(sample.keys())}")
                
            self.image_data.append(sample)
            
            # Extract all captions for this image
            captions = []
            if 'sentences' in sample:
                sents = sample['sentences']
                if isinstance(sents, list):
                    captions = [ (s['raw'] if isinstance(s, dict) and 'raw' in s else str(s)) for s in sents ]
                elif isinstance(sents, dict) and 'raw' in sents:
                    raw = sents['raw']
                    captions = raw if isinstance(raw, list) else [raw]
            elif 'captions' in sample:
                captions = sample['captions']
            elif 'caption' in sample:
                captions = sample['caption']
            
            # Ensure we have exactly 5 captions (pad or truncate)
            if len(captions) < 5:
                # Repeat captions to reach 5 (common practice)
                while len(captions) < 5:
                    print(f"Padding captions for image {img_idx} with {len(captions)} captions")
                    captions.extend(captions[:5-len(captions)])
            captions = captions[:5]  # Take first 5
            
            # Store caption mapping
            caption_indices = []
            for caption in captions:
                self.caption_data.append(caption)
                self.caption_to_image[caption_idx] = img_idx
                caption_indices.append(caption_idx)
                caption_idx += 1
                
                # For single caption mode, only use first caption
                if not use_all_captions:
                    break
            
            self.image_to_captions[img_idx] = caption_indices
        
        print(f"Loaded {len(self.image_data)} images and {len(self.caption_data)} captions")
        print(f"Average captions per image: {len(self.caption_data) / len(self.image_data):.1f}")
    
    def __len__(self):
        # Return number of images for iteration
        return len(self.image_data)
    
    def get_num_captions(self):
        return len(self.caption_data)
    
    def get_image_by_idx(self, img_idx):
        """Get image by image index"""
        sample = self.image_data[img_idx]
        
        # Handle different dataset formats
        if 'image' in sample:
            image = sample['image']
        elif 'jpg' in sample:
            image_data = sample["jpg"]
            image = image_data.convert("RGB") if hasattr(image_data, "convert") else Image.open(io.BytesIO(image_data)).convert("RGB")
        elif self.local_image_dir and 'filename' in sample:
            # ✅ 直接用 filename + .jpg，这样就和 download_images.py 保存的文件对应了
            filename = sample['filename']
            local_path = os.path.join(self.local_image_dir, f"{filename}")
            
            try:
                image = Image.open(local_path).convert("RGB")
            except FileNotFoundError:
                print(f"⚠️  本地图片不存在: {local_path}，尝试从 URL 下载...")
                if 'url' in sample:
                    try:
                        response = requests.get(sample['url'], timeout=10)
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    except Exception as e:
                        print(f"Error loading image from URL: {e}")
                        image = Image.new('RGB', (224, 224), color='black')
                else:
                    image = Image.new('RGB', (224, 224), color='black')
            except Exception as e:
                print(f"Error loading image from local path {local_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        elif 'url' in sample:
            # For COCO Karpathy: load image from URL
            if img_idx == 0:
                print(f"⚠️  未找到本地图片目录或 image_id 字段，使用 URL 下载")
                print(f"   - local_image_dir: {self.local_image_dir}")
                print(f"   - 'image_id' in sample: {'image_id' in sample}")
                print(f"   - sample 字段: {list(sample.keys())}")
            try:
                response = requests.get(sample['url'], timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as e:
                print(f"Error loading image from URL: {e}")
                # Create a blank image as fallback
                image = Image.new('RGB', (224, 224), color='black')
        else:
            raise ValueError("No image field found in dataset")
        
        return image
    
    def get_caption_by_idx(self, cap_idx):
        """Get caption by caption index"""
        return self.caption_data[cap_idx]
    
    def __getitem__(self, idx):
        # This is used for the DataLoader - return image and its first caption
        image = self.get_image_by_idx(idx)
        caption = self.caption_data[self.image_to_captions[idx][0]]  # First caption
        return image, caption, idx


def collate_retrieval_fn(batch, processor):
    """Collate function for retrieval evaluation (CPU only, move to device later)"""
    images, captions, indices = zip(*batch)
    
    # Process images (keep on CPU)
    image_inputs = processor(images=list(images), return_tensors="pt")
    
    # Set text max length based on processor type
    proc_name = processor.__class__.__name__.lower()
    text_max_len = 64 if "siglip" in proc_name else 77
    
    # Process captions (keep on CPU)
    text_inputs = processor(text=list(captions), return_tensors="pt", padding="max_length", truncation=True, max_length=text_max_len)
    
    indices = torch.tensor(indices)
    
    return image_inputs, text_inputs, indices


def compute_retrieval_metrics_karpathy(image_features, text_features, dataset, device=None):
    """
    Compute Karpathy-style retrieval metrics with 5 captions per image
    
    Args:
        image_features: [N_images, D] normalized image features (can be on GPU)
        text_features: [N_captions, D] normalized text features (can be on GPU, 5x more than images)
        dataset: RetrievalDataset instance with mapping info
        device: device to use for computation (if None, use same device as features)
    
    Returns:
        Dictionary with I2T and T2I metrics following Karpathy evaluation
    """
    # Determine device
    if device is None:
        device = image_features.device
    
    # Move features to device if needed
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    
    num_images = len(image_features)
    num_captions = len(text_features)
    
    print(f"Computing Karpathy metrics on {device}: {num_images} images, {num_captions} captions")
    
    # Compute similarity matrix [N_images, N_captions] on GPU
    print("Computing similarity matrix...")
    sim_matrix = image_features @ text_features.T
    
    # Image-to-Text retrieval (each image queries all captions)
    print("Computing Image-to-Text ranks...")
    i2t_ranks = []
    # Batch process for efficiency
    batch_size_i2t = 1000  # Process images in batches
    for batch_start in tqdm(range(0, num_images, batch_size_i2t), desc="I2T ranks"):
        batch_end = min(batch_start + batch_size_i2t, num_images)
        batch_sims = sim_matrix[batch_start:batch_end]  # [batch_size, N_captions]
        
        # Sort all images in batch at once (on GPU)
        sorted_indices_batch = torch.argsort(batch_sims, dim=1, descending=True)  # [batch_size, N_captions]
        
        for local_idx, img_idx in enumerate(range(batch_start, batch_end)):
            sorted_indices = sorted_indices_batch[local_idx]
            # Find ranks of the 5 ground-truth captions for this image
            gt_caption_indices = dataset.image_to_captions[img_idx]
            ranks = []
            for cap_idx in gt_caption_indices:
                rank = torch.where(sorted_indices == cap_idx)[0][0].item() + 1  # 1-indexed
                ranks.append(rank)
            
            # Use the best (minimum) rank among the 5 captions (standard practice)
            best_rank = min(ranks)
            i2t_ranks.append(best_rank)
    
    # Text-to-Image retrieval (each caption queries all images)
    print("Computing Text-to-Image ranks...")
    t2i_ranks = []
    # Batch process for efficiency
    batch_size_t2i = 5000  # Process captions in batches (can be larger since fewer operations per caption)
    for batch_start in tqdm(range(0, num_captions, batch_size_t2i), desc="T2I ranks"):
        batch_end = min(batch_start + batch_size_t2i, num_captions)
        batch_sims = sim_matrix[:, batch_start:batch_end]  # [N_images, batch_size]
        
        # Sort all captions in batch at once (on GPU)
        sorted_indices_batch = torch.argsort(batch_sims, dim=0, descending=True)  # [N_images, batch_size]
        
        for local_idx, cap_idx in enumerate(range(batch_start, batch_end)):
            sorted_indices = sorted_indices_batch[:, local_idx]
            # Find rank of the ground-truth image for this caption
            gt_img_idx = dataset.caption_to_image[cap_idx]
            rank = torch.where(sorted_indices == gt_img_idx)[0][0].item() + 1  # 1-indexed
            t2i_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    # Compute standard metrics
    metrics = {
        # Image-to-Text (5K images)
        "i2t_r1": np.mean(i2t_ranks <= 1) * 100,
        "i2t_r5": np.mean(i2t_ranks <= 5) * 100,
        "i2t_r10": np.mean(i2t_ranks <= 10) * 100,
        "i2t_mean_rank": np.mean(i2t_ranks),
        "i2t_median_rank": np.median(i2t_ranks),
        
        # Text-to-Image (25K captions)  
        "t2i_r1": np.mean(t2i_ranks <= 1) * 100,
        "t2i_r5": np.mean(t2i_ranks <= 5) * 100,
        "t2i_r10": np.mean(t2i_ranks <= 10) * 100,
        "t2i_mean_rank": np.mean(t2i_ranks),
        "t2i_median_rank": np.median(t2i_ranks),
    }
    
    # Compute average metrics (standard in papers)
    metrics["avg_r1"] = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2
    metrics["avg_r5"] = (metrics["i2t_r5"] + metrics["t2i_r5"]) / 2
    metrics["avg_r10"] = (metrics["i2t_r10"] + metrics["t2i_r10"]) / 2
    
    return metrics


def compute_retrieval_metrics(image_features, text_features, return_ranks=False, device=None):
    """
    Compute standard retrieval metrics (1:1 image-caption pairs)
    
    Args:
        image_features: [N, D] normalized image features (can be on GPU)
        text_features: [N, D] normalized text features (can be on GPU)
        return_ranks: whether to return individual ranks
        device: device to use for computation (if None, use same device as features)
    
    Returns:
        Dictionary with I2T and T2I metrics
    """
    # Determine device
    if device is None:
        device = image_features.device
    
    # Move features to device if needed
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    
    # Compute similarity matrix on GPU
    print("Computing similarity matrix...")
    sim_matrix = image_features @ text_features.T  # [N, N]
    
    # Image-to-Text retrieval
    print("Computing Image-to-Text ranks...")
    i2t_ranks = []
    # Batch process for efficiency
    batch_size_i2t = 1000
    for batch_start in tqdm(range(0, len(image_features), batch_size_i2t), desc="I2T ranks"):
        batch_end = min(batch_start + batch_size_i2t, len(image_features))
        batch_sims = sim_matrix[batch_start:batch_end]  # [batch_size, N]
        
        # Sort all images in batch at once (on GPU)
        sorted_indices_batch = torch.argsort(batch_sims, dim=1, descending=True)  # [batch_size, N]
        
        for local_idx, i in enumerate(range(batch_start, batch_end)):
            sorted_indices = sorted_indices_batch[local_idx]
            # Find rank of correct text (index i)
            rank = torch.where(sorted_indices == i)[0][0].item() + 1  # 1-indexed
            i2t_ranks.append(rank)
    
    # Text-to-Image retrieval  
    print("Computing Text-to-Image ranks...")
    t2i_ranks = []
    # Batch process for efficiency
    batch_size_t2i = 1000
    for batch_start in tqdm(range(0, len(text_features), batch_size_t2i), desc="T2I ranks"):
        batch_end = min(batch_start + batch_size_t2i, len(text_features))
        batch_sims = sim_matrix[:, batch_start:batch_end]  # [N, batch_size]
        
        # Sort all texts in batch at once (on GPU)
        sorted_indices_batch = torch.argsort(batch_sims, dim=0, descending=True)  # [N, batch_size]
        
        for local_idx, i in enumerate(range(batch_start, batch_end)):
            sorted_indices = sorted_indices_batch[:, local_idx]
            # Find rank of correct image (index i)
            rank = torch.where(sorted_indices == i)[0][0].item() + 1  # 1-indexed
            t2i_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    # Compute standard metrics
    metrics = {
        # Image-to-Text
        "i2t_r1": np.mean(i2t_ranks <= 1) * 100,
        "i2t_r5": np.mean(i2t_ranks <= 5) * 100, 
        "i2t_r10": np.mean(i2t_ranks <= 10) * 100,
        "i2t_mean_rank": np.mean(i2t_ranks),
        "i2t_median_rank": np.median(i2t_ranks),
        
        # Text-to-Image
        "t2i_r1": np.mean(t2i_ranks <= 1) * 100,
        "t2i_r5": np.mean(t2i_ranks <= 5) * 100,
        "t2i_r10": np.mean(t2i_ranks <= 10) * 100, 
        "t2i_mean_rank": np.mean(t2i_ranks),
        "t2i_median_rank": np.median(t2i_ranks),
    }
    
    # Compute average metrics (standard in papers)
    metrics["avg_r1"] = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2
    metrics["avg_r5"] = (metrics["i2t_r5"] + metrics["t2i_r5"]) / 2  
    metrics["avg_r10"] = (metrics["i2t_r10"] + metrics["t2i_r10"]) / 2
    
    if return_ranks:
        return metrics, i2t_ranks, t2i_ranks
    return metrics


def run_retrieval_evaluation(
    model_id="google/siglip2-so400m-patch14-384",
    model_type="clip",  # "clip" or "siglip"
    dataset_name="mscoco", 
    split="test",
    batch_size=64,
    num_workers=8,
    max_samples=None,
    save_features=False,
    device=None,
    use_karpathy_eval=True,  # Use 5-caption Karpathy evaluation
    local_image_dir=None  # ✅ 本地图片目录
):
    """
    Run CLIP/SigLIP retrieval evaluation
    
    Args:
        model_id: HuggingFace model ID or local path
        model_type: "clip" or "siglip" - determines which model/processor to use
        dataset_name: "mscoco" or "flickr30k"
        split: dataset split ("test", "validation")
        batch_size: batch size for inference
        num_workers: (deprecated) number of data loading workers - set to 0 to avoid torch pickling issues
        max_samples: limit number of samples (for debugging)
        save_features: save computed features for analysis
        device: device to use ("cuda", "cuda:0", "cpu", etc.). If None, auto-detect
        use_karpathy_eval: Use 5-caption Karpathy evaluation
        local_image_dir: ✅ 本地图片目录，如果指定则优先从本地加载
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect model type if not specified or if "auto"
    if model_type == "auto" or model_type is None:
        model_id_lower = model_id.lower()
        if "siglip" in model_id_lower:
            model_type = "siglip"
        elif "clip" in model_id_lower or "openai" in model_id_lower:
            model_type = "clip"
        else:
            # Default to CLIP if can't determine
            model_type = "clip"
            print(f"⚠️  Could not auto-detect model type from '{model_id}', defaulting to CLIP")
    
    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    print(f"Model type: {model_type.upper()}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    # Load model and processor based on model type
    print(f"Loading {model_type.upper()} model and processor...")
    if model_type.lower() == "clip":
        model = AutoModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif model_type.lower() == "siglip":
        model = SiglipModel.from_pretrained(model_id)
        # processor = SiglipProcessor.from_pretrained("/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384")
        processor = SiglipProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'clip' or 'siglip'")
    
    model.to(device).eval()
    
    # Load dataset using Karpathy splits (standard for retrieval evaluation)
    print("Loading dataset...")
    if dataset_name.lower() == "mscoco":
        print("📥 Using COCO Karpathy split - standard for retrieval evaluation")
        # Karpathy split: validation (5K) and test (5K) - perfect for standard evaluation
        # ds = load_dataset("/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/coco-karpathy", split=split)
        ds = load_dataset("yerevann/coco-karpathy", split="validation")
        print(f"✅ Loaded COCO Karpathy {split} split (5K samples)")
    elif dataset_name.lower() == "flickr30k":
        ds = load_dataset("nlphuji/flickr30k", split="test")
        print(f"📊 Flickr30K test split loaded: {len(ds)} samples")
        # 检查是否有其他splits可用
        try:
            all_splits = load_dataset("nlphuji/flickr30k")
            print(f"📊 Available splits: {list(all_splits.keys())}")
            for split_name in all_splits.keys():
                print(f"   - {split_name}: {len(all_splits[split_name])} samples")
        except:
            pass
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create dataset and dataloader
    print(f"📥 Raw dataset size (before max_samples): {len(ds)}")
    if max_samples:
        print(f"⚠️  max_samples={max_samples} will limit the dataset")
    dataset = RetrievalDataset(ds, processor, split, max_samples, use_all_captions=use_karpathy_eval, local_image_dir=local_image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # ✅ 修改为 0 避免 pickle 序列化问题
        pin_memory=True,
        collate_fn=lambda batch: collate_retrieval_fn(batch, processor)
    )
    
    print(f"✅ Final dataset size: {len(dataset)} images, {dataset.get_num_captions()} captions")
    
    # Extract features
    all_image_features = []
    all_text_features = []
    
    print("Extracting image features...")
    # First pass: extract image features
    with torch.no_grad():
        for image_inputs, text_inputs, indices in tqdm(dataloader, desc="Processing image batches"):
            # Move data to device in main process (not in workers)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_image_features.append(image_features.cpu())
    
    # Concatenate image features (keep on GPU)
    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    print(f"Image features shape: {all_image_features.shape} (on {device})")
    
    # Second pass: extract ALL caption features (if using Karpathy evaluation)
    if use_karpathy_eval:
        print("Extracting ALL caption features (5 per image)...")
        # Process all captions in batches
        caption_batch_size = batch_size * 2  # Can use larger batch for text
        num_captions = dataset.get_num_captions()
        
        for start_idx in tqdm(range(0, num_captions, caption_batch_size), desc="Processing caption batches"):
            end_idx = min(start_idx + caption_batch_size, num_captions)
            batch_captions = [dataset.get_caption_by_idx(i) for i in range(start_idx, end_idx)]
            
            # Process caption batch
            proc_name = processor.__class__.__name__.lower()
            text_max_len = 64 if "siglip" in proc_name else 77

            text_inputs = processor(text=batch_captions, return_tensors="pt", padding="max_length", truncation=True, max_length=text_max_len)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_text_features.append(text_features)  # Keep on GPU
        
        # Concatenate all text features (keep on GPU)
        all_text_features = torch.cat(all_text_features, dim=0).to(device)
    else:
        # Standard evaluation: use only first caption per image
        print("Extracting text features (1 per image)...")
        with torch.no_grad():
            for image_inputs, text_inputs, indices in tqdm(dataloader, desc="Processing text batches"):
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_text_features.append(text_features)  # Keep on GPU
        
        all_text_features = torch.cat(all_text_features, dim=0).to(device)
    
    print(f"Text features shape: {all_text_features.shape} (on {device})")
    
    # Compute retrieval metrics (on GPU for speed)
    print("Computing retrieval metrics...")
    if use_karpathy_eval:
        print("Using Karpathy-style evaluation (5 captions per image)")
        metrics = compute_retrieval_metrics_karpathy(all_image_features, all_text_features, dataset, device=device)
    else:
        print("Using standard 1:1 evaluation")
        metrics = compute_retrieval_metrics(all_image_features, all_text_features, device=device)
    
    # Print results in standard format
    eval_mode = "Karpathy (5 captions/image)" if use_karpathy_eval else "Standard (1:1)"
    print("\n" + "="*70)
    print(f"RETRIEVAL RESULTS - {dataset_name.upper()} {split.upper()}")
    print(f"Evaluation Mode: {eval_mode}")
    print(f"Images: {len(all_image_features)}, Captions: {len(all_text_features)}")
    print("="*70)
    print("Image-to-Text Retrieval:")
    print(f"  R@1:  {metrics['i2t_r1']:.2f}%")
    print(f"  R@5:  {metrics['i2t_r5']:.2f}%") 
    print(f"  R@10: {metrics['i2t_r10']:.2f}%")
    print(f"  Mean Rank: {metrics['i2t_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['i2t_median_rank']:.1f}")
    
    print("\nText-to-Image Retrieval:")
    print(f"  R@1:  {metrics['t2i_r1']:.2f}%")
    print(f"  R@5:  {metrics['t2i_r5']:.2f}%")
    print(f"  R@10: {metrics['t2i_r10']:.2f}%") 
    print(f"  Mean Rank: {metrics['t2i_mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['t2i_median_rank']:.1f}")
    
    print("\nAverage Metrics:")
    print(f"  Avg R@1:  {metrics['avg_r1']:.2f}%")
    print(f"  Avg R@5:  {metrics['avg_r5']:.2f}%") 
    print(f"  Avg R@10: {metrics['avg_r10']:.2f}%")
    print("="*70)
    
    # Save results
    result_info = {
        "model": model_id,
        "model_type": model_type,
        "dataset": dataset_name,
        "split": split,
        "evaluation_mode": "karpathy" if use_karpathy_eval else "standard",
        "num_images": len(all_image_features),
        "num_captions": len(all_text_features),
        "captions_per_image": len(all_text_features) / len(all_image_features),
        "batch_size": batch_size,
        **metrics
    }
    
    # Create results directory if it doesn't exist
    results_dir = "/home/muzammal/Projects/CLIP-R/eval/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    model_name = model_id.replace("/", "_")
    eval_suffix = "_karpathy" if use_karpathy_eval else "_standard"
    
    # Save text results
    text_file = os.path.join(results_dir, f"retrieval_results_{model_type}_{model_name}_{dataset_name}_{split}{eval_suffix}.txt")
    with open(text_file, "w") as f:
        f.write(f"RETRIEVAL RESULTS - {dataset_name.upper()} {split.upper()}\n")
        f.write(f"Evaluation Mode: {eval_mode}\n")
        f.write(f"Images: {len(all_image_features)}, Captions: {len(all_text_features)}\n")
        f.write("="*70 + "\n")
        f.write("Image-to-Text Retrieval:\n")
        f.write(f"  R@1:  {metrics['i2t_r1']:.2f}%\n")
        f.write(f"  R@5:  {metrics['i2t_r5']:.2f}%\n")
        f.write(f"  R@10: {metrics['i2t_r10']:.2f}%\n")
        f.write(f"  Mean Rank: {metrics['i2t_mean_rank']:.2f}\n")
        f.write(f"  Median Rank: {metrics['i2t_median_rank']:.1f}\n")
        f.write("\nText-to-Image Retrieval:\n")
        f.write(f"  R@1:  {metrics['t2i_r1']:.2f}%\n")
        f.write(f"  R@5:  {metrics['t2i_r5']:.2f}%\n") 
        f.write(f"  R@10: {metrics['t2i_r10']:.2f}%\n")
        f.write(f"  Mean Rank: {metrics['t2i_mean_rank']:.2f}\n")
        f.write(f"  Median Rank: {metrics['t2i_median_rank']:.1f}\n")
        f.write("\nAverage Metrics:\n")
        f.write(f"  Avg R@1:  {metrics['avg_r1']:.2f}%\n")
        f.write(f"  Avg R@5:  {metrics['avg_r5']:.2f}%\n")
        f.write(f"  Avg R@10: {metrics['avg_r10']:.2f}%\n")
    
    
    return result_info


if __name__ == "__main__":
    # SigLIP-R evaluation on MSCOCO Karpathy split (optimized for 64GB VRAM)
    print("Running MSCOCO 5K retrieval evaluation (Karpathy 5-caption mode)...")
    print("Model: SigLIP-R (trained with CLIP-R method)")
    
    # 修改这里的模型路径为你训练好的 SigLIP-R 模型
    # MODEL_PATH = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
    MODEL_PATH = "fesvhtr/siglip-r-s1-run1027-1536"

    # coco_results = run_retrieval_evaluation(
    #     model_id=MODEL_PATH,
    #     model_type="clip",  # SigLIP model
    #     dataset_name="mscoco",
    #     split="test",  # Karpathy test split (5K samples)
    #     batch_size=128,  # 大batch size，充分利用64GB显存
    #     max_samples=None,  # Use full Karpathy split (exactly 5K)
    #     device="cuda:2",
    #     use_karpathy_eval=True,  # 🔥 Enable standard Karpathy 5-caption evaluation
    #     local_image_dir="/home/muzammal/Projects/CLIP-R/data/coco_images" # ✅ 改为正确的下载目录
    # )
    
    
    # Also test on Flickr30K
    print("\n" + "="*80)
    print("Running Flickr30K retrieval evaluation...")
    flickr_results = run_retrieval_evaluation(
        model_id=MODEL_PATH,
        model_type="siglip",  # 明确指定模型类型
        dataset_name="flickr30k",
        split="test",
        batch_size=384,
        max_samples=None,  # 使用全部数据（不限制）
        device="cuda:1",
        use_karpathy_eval=True,  # 使用 5-caption Karpathy 评估
        # local_image_dir="/path/to/flickr30k/images"  # 如果需要本地图片目录，取消注释并设置路径
    ) 