import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
import json
import os
import requests


class RetrievalDataset(torch.utils.data.Dataset):
    """Standard retrieval dataset for COCO/Flickr30K with Karpathy 5-caption support"""
    def __init__(self, dataset, processor, split="test", max_samples=None, use_all_captions=True):
        self.processor = processor
        self.use_all_captions = use_all_captions
        self.image_data = []  # Store image info
        self.caption_data = []  # Store all captions
        self.image_to_captions = {}  # Map image index to caption indices
        self.caption_to_image = {}  # Map caption index to image index
        
        print(f"Loading {split} split...")
        print(f"Karpathy evaluation mode: {'5 captions per image' if use_all_captions else '1 caption per image'}")
        
        caption_idx = 0
        for img_idx, sample in enumerate(tqdm(dataset)):
            if max_samples and img_idx >= max_samples:
                break
                
            self.image_data.append(sample)
            
            # Extract all captions for this image
            captions = []
            if 'sentences' in sample:
                # For COCO Karpathy: extract all captions from sentences
                if isinstance(sample['sentences'], list):
                    captions = sample['sentences']
                elif 'raw' in sample['sentences']:
                    captions = sample['sentences']['raw']
            elif 'captions' in sample:
                captions = sample['captions']
            elif 'caption' in sample:
                captions = [sample['caption']]
            
            # Ensure we have exactly 5 captions (pad or truncate)
            if len(captions) < 5:
                # Repeat captions to reach 5 (common practice)
                while len(captions) < 5:
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
        elif 'url' in sample:
            # For COCO Karpathy: load image from URL
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
    
    # Process captions (keep on CPU)
    text_inputs = processor(text=list(captions), return_tensors="pt", padding=True, truncation=True, max_length=77)
    
    indices = torch.tensor(indices)
    
    return image_inputs, text_inputs, indices


def compute_retrieval_metrics_karpathy(image_features, text_features, dataset):
    """
    Compute Karpathy-style retrieval metrics with 5 captions per image
    
    Args:
        image_features: [N_images, D] normalized image features  
        text_features: [N_captions, D] normalized text features (5x more than images)
        dataset: RetrievalDataset instance with mapping info
    
    Returns:
        Dictionary with I2T and T2I metrics following Karpathy evaluation
    """
    num_images = len(image_features)
    num_captions = len(text_features)
    
    print(f"Computing Karpathy metrics: {num_images} images, {num_captions} captions")
    
    # Compute similarity matrix [N_images, N_captions]
    sim_matrix = image_features @ text_features.T
    
    # Image-to-Text retrieval (each image queries all captions)
    i2t_ranks = []
    for img_idx in range(num_images):
        # Get similarities for this image with all captions
        sims = sim_matrix[img_idx]  # [N_captions]
        # Sort in descending order and get indices
        sorted_indices = torch.argsort(sims, descending=True)
        
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
    t2i_ranks = []
    for cap_idx in range(num_captions):
        # Get similarities for this caption with all images  
        sims = sim_matrix[:, cap_idx]  # [N_images]
        # Sort in descending order and get indices
        sorted_indices = torch.argsort(sims, descending=True)
        
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


def compute_retrieval_metrics(image_features, text_features, return_ranks=False):
    """
    Compute standard retrieval metrics (1:1 image-caption pairs)
    
    Args:
        image_features: [N, D] normalized image features
        text_features: [N, D] normalized text features
        return_ranks: whether to return individual ranks
    
    Returns:
        Dictionary with I2T and T2I metrics
    """
    # Compute similarity matrix
    sim_matrix = image_features @ text_features.T  # [N, N]
    
    # Image-to-Text retrieval
    i2t_ranks = []
    for i in range(len(image_features)):
        # Get similarities for this image with all texts
        sims = sim_matrix[i]  # [N]
        # Sort in descending order and get ranks
        sorted_indices = torch.argsort(sims, descending=True)
        # Find rank of correct text (index i)
        rank = torch.where(sorted_indices == i)[0][0].item() + 1  # 1-indexed
        i2t_ranks.append(rank)
    
    # Text-to-Image retrieval  
    t2i_ranks = []
    for i in range(len(text_features)):
        # Get similarities for this text with all images
        sims = sim_matrix[:, i]  # [N]
        # Sort in descending order and get ranks
        sorted_indices = torch.argsort(sims, descending=True)
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
    model_id="openai/clip-vit-base-patch32",
    dataset_name="mscoco", 
    split="test",
    batch_size=64,
    num_workers=8,
    use_amp=True,
    max_samples=None,
    save_features=False,
    device=None,
    use_karpathy_eval=True  # Use 5-caption Karpathy evaluation
):
    """
    Run standard CLIP retrieval evaluation
    
    Args:
        model_id: HuggingFace model ID or local path
        dataset_name: "mscoco" or "flickr30k"
        split: dataset split ("test", "validation")
        batch_size: batch size for inference
        num_workers: number of data loading workers
        use_amp: use automatic mixed precision
        max_samples: limit number of samples (for debugging)
        save_features: save computed features for analysis
        device: device to use ("cuda", "cuda:0", "cpu", etc.). If None, auto-detect
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Batch size: {batch_size}, Workers: {num_workers}, AMP: {use_amp}")
    
    # Load model and processor
    print("Loading model and processor...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.to(device).eval()
    
    # Load dataset using Karpathy splits (standard for retrieval evaluation)
    print("Loading dataset...")
    if dataset_name.lower() == "mscoco":
        print("📥 Using COCO Karpathy split - standard for retrieval evaluation")
        # Karpathy split: validation (5K) and test (5K) - perfect for standard evaluation
        ds = load_dataset("yerevann/coco-karpathy", split=split)
        print(f"✅ Loaded COCO Karpathy {split} split (5K samples)")
    elif dataset_name.lower() == "flickr30k":
        ds = load_dataset("nlphuji/flickr30k", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create dataset and dataloader
    dataset = RetrievalDataset(ds, processor, split, max_samples, use_all_captions=use_karpathy_eval)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_retrieval_fn(batch, processor)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Extract features
    all_image_features = []
    all_text_features = []
    
    print("Extracting image features...")
    # First pass: extract image features
    with torch.no_grad():
        for image_inputs, text_inputs, indices in tqdm(dataloader, desc="Processing image batches"):
            # Move data to device in main process (not in workers)
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    image_features = model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_image_features.append(image_features.cpu())
    
    # Concatenate image features
    all_image_features = torch.cat(all_image_features, dim=0)
    print(f"Image features shape: {all_image_features.shape}")
    
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
            text_inputs = processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        text_features = model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                else:
                    text_features = model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_text_features.append(text_features.cpu())
        
        # Concatenate all text features
        all_text_features = torch.cat(all_text_features, dim=0)
    else:
        # Standard evaluation: use only first caption per image
        print("Extracting text features (1 per image)...")
        with torch.no_grad():
            for image_inputs, text_inputs, indices in tqdm(dataloader, desc="Processing text batches"):
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        text_features = model.get_text_features(**text_inputs)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                else:
                    text_features = model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                all_text_features.append(text_features.cpu())
        
        all_text_features = torch.cat(all_text_features, dim=0)
    
    print(f"Text features shape: {all_text_features.shape}")
    
    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    if use_karpathy_eval:
        print("Using Karpathy-style evaluation (5 captions per image)")
        metrics = compute_retrieval_metrics_karpathy(all_image_features, all_text_features, dataset)
    else:
        print("Using standard 1:1 evaluation")
        metrics = compute_retrieval_metrics(all_image_features, all_text_features)
    
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
        "dataset": dataset_name,
        "split": split,
        "evaluation_mode": "karpathy" if use_karpathy_eval else "standard",
        "num_images": len(all_image_features),
        "num_captions": len(all_text_features),
        "captions_per_image": len(all_text_features) / len(all_image_features),
        "batch_size": batch_size,
        "amp_enabled": use_amp,
        **metrics
    }
    
    # Create results directory if it doesn't exist
    results_dir = "/home/muzammal/Projects/CLIP-R/eval/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    model_name = model_id.replace("/", "_")
    eval_suffix = "_karpathy" if use_karpathy_eval else "_standard"
    results_file = os.path.join(results_dir, f"retrieval_results_{model_name}_{dataset_name}_{split}{eval_suffix}.json")
    with open(results_file, "w") as f:
        json.dump(result_info, f, indent=2)
    
    # Save text results
    text_file = os.path.join(results_dir, f"retrieval_results_{model_name}_{dataset_name}_{split}{eval_suffix}.txt")
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
    
    # Optionally save features for further analysis
    if save_features:
        features_file = os.path.join(results_dir, f"features_{model_name}_{dataset_name}_{split}{eval_suffix}.npz")
        np.savez(features_file, 
                image_features=all_image_features.numpy(),
                text_features=all_text_features.numpy())
        print(f"Features saved to: {features_file}")
    
    print(f"Results saved to: {results_file}")
    return result_info


if __name__ == "__main__":
    # Standard CLIP evaluation on MSCOCO Karpathy split
    print("Running MSCOCO 5K retrieval evaluation (Karpathy 5-caption mode)...")
    coco_results = run_retrieval_evaluation(
        model_id="fesvhtr/clip_r_best_model_demo_0621_192211",
        dataset_name="mscoco",
        split="test",  # Karpathy validation split (5K samples)
        batch_size=16,
        num_workers=16,
        use_amp=False,
        max_samples=None,  # Use full Karpathy split (exactly 5K)
        device="cuda:0",  # 可以指定具体的GPU
        use_karpathy_eval=True  # 🔥 Enable standard Karpathy 5-caption evaluation
    )
    
    print(f"\n🎯 Karpathy Evaluation Results:")
    print(f"Image→Text: R@1={coco_results['i2t_r1']:.2f}%, R@5={coco_results['i2t_r5']:.2f}%, R@10={coco_results['i2t_r10']:.2f}%")
    print(f"Text→Image: R@1={coco_results['t2i_r1']:.2f}%, R@5={coco_results['t2i_r5']:.2f}%, R@10={coco_results['t2i_r10']:.2f}%")
    print(f"Average: R@1={coco_results['avg_r1']:.2f}%, R@5={coco_results['avg_r5']:.2f}%, R@10={coco_results['avg_r10']:.2f}%")
    
    # Also test on Flickr30K
    # print("\n" + "="*80)
    # print("Running Flickr30K 1K retrieval evaluation...")
    # flickr_results = run_retrieval_evaluation(
    #     model_id="openai/clip-vit-base-patch32", 
    #     dataset_name="flickr30k",
    #     split="test",
    #     batch_size=64,
    #     num_workers=8,
    #     use_amp=True,
    #     max_samples=1000  # Standard 1K evaluation
    # ) 