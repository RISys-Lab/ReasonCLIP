"""
Zero-shot ImageNet classification evaluation
Supports: ImageNet-1K, ImageNet-V2, ObjectNet
Same logic as CLIP_benchmark
"""
import torch
import os
import argparse
import glob
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset configs: (name, hf_id, split)
DATASETS_CONFIG = {
    "imagenet1k": ("clip-benchmark/wds_imagenet1k", "test"),
    "imagenetv2": ("clip-benchmark/wds_imagenetv2", "test"),
    "objectnet": ("clip-benchmark/wds_objectnet", "test"),
}


def _read_txt_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _get_hf_home(hf_home=None):
    if hf_home:
        return hf_home
    return os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


def _find_cached_dataset_file(hf_id, filename, hf_home=None):
    hf_home = _get_hf_home(hf_home)
    repo_dir = f"datasets--{hf_id.replace('/', '--')}"
    hub_dir = os.path.join(hf_home, "hub", repo_dir)
    if not os.path.isdir(hub_dir):
        return None
    snapshots_dir = os.path.join(hub_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    candidates = glob.glob(os.path.join(snapshots_dir, "*", filename))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_wds_metadata(hf_id, classnames_file=None, templates_file=None, hf_home=None):
    """Load classnames and templates from local HF cache or provided txt paths."""
    if classnames_file is None:
        classnames_file = _find_cached_dataset_file(hf_id, "classnames.txt", hf_home=hf_home)
    if templates_file is None:
        templates_file = _find_cached_dataset_file(hf_id, "zeroshot_classification_templates.txt", hf_home=hf_home)

    if classnames_file is None or templates_file is None:
        raise FileNotFoundError(
            "Local metadata not found. Provide --classnames_file/--templates_file or set HF_HOME."
        )

    classnames = _read_txt_lines(classnames_file)
    templates = _read_txt_lines(templates_file)
    return classnames, templates


def create_text_features(classnames, templates, processor, model, device):
    """Create text features for all classes using prompt templates"""
    print(f"Computing text features for {len(classnames)} classes with {len(templates)} templates...")
    
    all_text_features = []
    
    for classname in tqdm(classnames, desc="Processing classes"):
        class_prompts = [template.format(c=classname) for template in templates]
        
        with torch.no_grad():
            inputs = processor(text=class_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Average over all templates
            class_text_feature = text_features.mean(dim=0, keepdim=True)
            class_text_feature = class_text_feature / class_text_feature.norm(dim=-1, keepdim=True)
            
            all_text_features.append(class_text_feature)
    
    text_features = torch.cat(all_text_features, dim=0)
    print(f"✅ Text features shape: {text_features.shape}")
    return text_features


class ZeroShotDataset(torch.utils.data.Dataset):
    """Dataset for zero-shot classification"""
    def __init__(self, hf_dataset, processor, dataset_name="imagenet1k", max_samples=None):
        self.processor = processor
        self.dataset_name = dataset_name
        self.data_list = []
        
        print(f"Loading {dataset_name} dataset...")
        
        for i, sample in enumerate(tqdm(hf_dataset, desc="Loading samples")):
            if max_samples and i >= max_samples:
                break
            self.data_list.append(sample)
        
        print(f"✅ Loaded {len(self.data_list)} samples")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Get image
        image_data = sample["png"]
        
        if hasattr(image_data, "convert"):
            img = image_data.convert("RGB")
        else:
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        label = int(sample["cls"])
        return img, label


def collate_fn(batch, processor):
    """Batch collate function"""
    images, labels = zip(*batch)
    image_inputs = processor(images=list(images), return_tensors="pt")
    labels = torch.tensor(labels)
    return image_inputs, labels


def run_zeroshot_evaluation(
    model_path,
    processor_path=None,
    dataset_name="imagenet1k",
    batch_size=64,
    num_workers=8,
    max_samples=None,
    device=None,
    results_dir=None,
    classnames_file=None,
    templates_file=None,
    hf_home=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if processor_path is None:
        processor_path = model_path
    
    hf_id, split = DATASETS_CONFIG[dataset_name]
    
    print("=" * 80)
    print("Zero-Shot Classification (CLIP_benchmark compatible)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} ({hf_id})")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load model and processor
    print(f"Loading model...")
    model = AutoModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    model.to(device).eval()
    
    # Load classnames and templates from dataset (same as CLIP_benchmark)
    print(f"Loading classnames and templates from local cache ({hf_id})...")
    classnames, templates = load_wds_metadata(
        hf_id,
        classnames_file=classnames_file,
        templates_file=templates_file,
        hf_home=hf_home,
    )
    print(f"📝 Loaded {len(classnames)} classes, {len(templates)} templates")
    
    # Create text features
    text_features = create_text_features(classnames, templates, processor, model, device)
    
    # Load dataset
    print(f"\n📥 Loading dataset...")
    hf_dataset = load_dataset(hf_id, split=split, streaming=True)
    dataset = ZeroShotDataset(hf_dataset, processor, dataset_name, max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    # Evaluation
    top1, top5, total = 0, 0, 0
    
    print("\n🚀 Starting evaluation...")
    with torch.no_grad():
        for image_inputs, labels in tqdm(dataloader, desc="Evaluating"):
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            labels = labels.to(device)
            
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = image_features @ text_features.T
            
            # Top-k (handle case where num_classes < 5)
            k = min(5, len(classnames))
            topk = logits.topk(k, dim=-1).indices
            
            top1 += (topk[:, 0] == labels).sum().item()
            if k >= 5:
                top5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    
    top1_acc = top1 / total * 100
    top5_acc = (top5 / total * 100) if k >= 5 else 0.0
    
    print("\n" + "=" * 80)
    print(f"📊 Results: {model_path} on {dataset_name}")
    print("=" * 80)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    if k >= 5:
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Total Samples: {total}")
    print(f"Num Classes: {len(classnames)}")
    print("=" * 80)
    
    # Save results
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = model_path.replace("/", "_")
    result_file = os.path.join(results_dir, f"zeroshot_{model_name}_{dataset_name}.txt")
    
    with open(result_file, "w") as f:
        f.write(f"Zero-Shot Classification Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"HF ID: {hf_id}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        if k >= 5:
            f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Num Classes: {len(classnames)}\n")
        f.write(f"Num Templates: {len(templates)}\n")
    
    print(f"💾 Results saved to: {result_file}")
    
    return {
        "model": model_path,
        "dataset": dataset_name,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_samples": total,
        "num_classes": len(classnames)
    }


def run_all_evaluations(
    model_path,
    processor_path=None,
    batch_size=64,
    num_workers=8,
    max_samples=None,
    device=None,
    results_dir=None,
    classnames_file=None,
    templates_file=None,
    hf_home=None
):
    """Run evaluation on all datasets"""
    from datetime import datetime
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if processor_path is None:
        processor_path = model_path
    
    print("=" * 80)
    print("🚀 Running ALL Zero-Shot Evaluations (CLIP_benchmark compatible)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Datasets: {list(DATASETS_CONFIG.keys())}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load model and processor once
    print(f"\nLoading model...")
    model = AutoModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    model.to(device).eval()
    
    all_results = []
    
    for dataset_name, (hf_id, split) in DATASETS_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"📊 Evaluating: {dataset_name}")
        print(f"{'='*80}")
        
        # Load classnames and templates for this dataset
        print(f"Loading classnames and templates from local cache ({hf_id})...")
        classnames, templates = load_wds_metadata(
            hf_id,
            classnames_file=classnames_file,
            templates_file=templates_file,
            hf_home=hf_home,
        )
        print(f"📝 {len(classnames)} classes, {len(templates)} templates")
        
        # Create text features for this dataset
        text_features = create_text_features(classnames, templates, processor, model, device)
        
        # Load dataset
        hf_dataset = load_dataset(hf_id, split=split, streaming=True)
        dataset = ZeroShotDataset(hf_dataset, processor, dataset_name, max_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=lambda batch: collate_fn(batch, processor)
        )
        
        # Evaluation
        top1, top5, total = 0, 0, 0
        k = min(5, len(classnames))
        
        with torch.no_grad():
            for image_inputs, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                image_inputs = {k_: v.to(device) for k_, v in image_inputs.items()}
                labels = labels.to(device)
                
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                topk = logits.topk(k, dim=-1).indices
                
                top1 += (topk[:, 0] == labels).sum().item()
                if k >= 5:
                    top5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)
        
        top1_acc = top1 / total * 100
        top5_acc = (top5 / total * 100) if k >= 5 else None
        
        result = {
            "dataset": dataset_name,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "total_samples": total,
            "num_classes": len(classnames)
        }
        all_results.append(result)
        
        if top5_acc is not None:
            print(f"✅ {dataset_name}: Top-1={top1_acc:.2f}%, Top-5={top5_acc:.2f}%, Classes={len(classnames)}")
        else:
            print(f"✅ {dataset_name}: Top-1={top1_acc:.2f}%, Classes={len(classnames)}")
    
    # Save combined results
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = model_path.replace("/", "_")
    result_file = os.path.join(results_dir, f"zeroshot_{model_name}_all.txt")
    
    with open(result_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Zero-Shot Classification Results (All Datasets)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Dataset':<15} {'Top-1':<10} {'Top-5':<10} {'Classes':<10} {'Samples':<10}\n")
        f.write("-" * 55 + "\n")
        
        for r in all_results:
            top5_str = f"{r['top5_accuracy']:.2f}" if r['top5_accuracy'] is not None else "N/A"
            f.write(f"{r['dataset']:<15} {r['top1_accuracy']:<10.2f} {top5_str:<10} {r['num_classes']:<10} {r['total_samples']:<10}\n")
        
        f.write("-" * 55 + "\n")
        
        # Average (only top1, since top5 may not be available for all)
        avg_top1 = sum(r['top1_accuracy'] for r in all_results) / len(all_results)
        f.write(f"{'Average':<15} {avg_top1:<10.2f}\n")
        f.write("=" * 70 + "\n")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Summary: All Zero-Shot Results")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Top-1':<10} {'Top-5':<10} {'Classes':<10} {'Samples':<10}")
    print("-" * 55)
    for r in all_results:
        top5_str = f"{r['top5_accuracy']:.2f}" if r['top5_accuracy'] is not None else "N/A"
        print(f"{r['dataset']:<15} {r['top1_accuracy']:<10.2f} {top5_str:<10} {r['num_classes']:<10} {r['total_samples']:<10}")
    print("-" * 55)
    print(f"{'Average Top-1':<15} {avg_top1:<10.2f}")
    print("=" * 80)
    print(f"💾 Results saved to: {result_file}")
    
    return {
        "model": model_path,
        "results": all_results,
        "avg_top1": avg_top1
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot classification (CLIP_benchmark compatible)")
    
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--processor_path", type=str, default=None, help="Processor path")
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                        choices=["imagenet1k", "imagenetv2", "objectnet", "all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--classnames_file", type=str, default=None)
    parser.add_argument("--templates_file", type=str, default=None)
    parser.add_argument("--hf_home", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.dataset == "all":
        result = run_all_evaluations(
            model_path=args.model_path,
            processor_path=args.processor_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            device=args.device,
            results_dir=args.results_dir,
            classnames_file=args.classnames_file,
            templates_file=args.templates_file,
            hf_home=args.hf_home
        )
        print(f"\n🎯 Final Average Top-1: {result['avg_top1']:.2f}%")
    else:
        result = run_zeroshot_evaluation(
            model_path=args.model_path,
            processor_path=args.processor_path,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            device=args.device,
            results_dir=args.results_dir,
            classnames_file=args.classnames_file,
            templates_file=args.templates_file,
            hf_home=args.hf_home
        )
        print(f"\n🎯 Final: Top-1={result['top1_accuracy']:.2f}%")
