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


def _local_metadata_paths(hf_id):
    dataset_dirname = hf_id.split("/")[-1]
    base_dir = os.path.join(SCRIPT_DIR, "eval_data", dataset_dirname)
    return (
        os.path.join(base_dir, "classnames.txt"),
        os.path.join(base_dir, "zeroshot_classification_templates.txt"),
    )


def _resolve_data_dir(hf_id, data_dir):
    if data_dir is None:
        return None
    dataset_dirname = hf_id.split("/")[-1]
    candidate = os.path.join(data_dir, dataset_dirname)
    if os.path.isdir(candidate):
        return candidate
    return data_dir


def _find_wds_files(data_dir):
    patterns = ["*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2", "*.wds", "*.webdataset", "*.shard"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))
    return sorted(set(files))


def _load_dataset_local_or_hub(hf_id, split, data_dir=None, streaming=True):
    resolved_dir = _resolve_data_dir(hf_id, data_dir)
    if resolved_dir is not None:
        split_dir = os.path.join(resolved_dir, split)
        data_dir_use = split_dir if os.path.isdir(split_dir) else resolved_dir
        wds_files = _find_wds_files(data_dir_use)
        if not wds_files:
            raise FileNotFoundError(
                f"No webdataset shards found under: {data_dir_use}. "
                "Expected *.tar/*.tar.gz/*.tgz/*.tar.bz2/*.wds/*.webdataset/*.shard"
            )
        data_files = {split: wds_files}
        return load_dataset("webdataset", data_files=data_files, split=split, streaming=streaming)
    return load_dataset(hf_id, split=split, streaming=streaming, local_files_only=True)


def load_wds_metadata(hf_id, classnames_file=None, templates_file=None):
    """Load classnames and templates from local eval_data or provided txt paths."""
    if classnames_file is None or templates_file is None:
        default_classnames, default_templates = _local_metadata_paths(hf_id)
        if classnames_file is None:
            classnames_file = default_classnames
        if templates_file is None:
            templates_file = default_templates

    if not os.path.isfile(classnames_file) or not os.path.isfile(templates_file):
        raise FileNotFoundError(
            "Local metadata not found. Provide --classnames_file/--templates_file or place "
            "txt files under eval/eval_data/<dataset>/."
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
    data_dir=None
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
    print(f"Loading classnames and templates from local files ({hf_id})...")
    classnames, templates = load_wds_metadata(
        hf_id,
        classnames_file=classnames_file,
        templates_file=templates_file,
    )
    print(f"📝 Loaded {len(classnames)} classes, {len(templates)} templates")
    
    # Create text features
    text_features = create_text_features(classnames, templates, processor, model, device)
    
    # Load dataset
    print(f"\n📥 Loading dataset...")
    hf_dataset = _load_dataset_local_or_hub(hf_id, split, data_dir=data_dir, streaming=True)
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
    data_dir=None
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
        print(f"Loading classnames and templates from local files ({hf_id})...")
        classnames, templates = load_wds_metadata(
            hf_id,
            classnames_file=classnames_file,
            templates_file=templates_file,
        )
        print(f"📝 {len(classnames)} classes, {len(templates)} templates")
        
        # Create text features for this dataset
        text_features = create_text_features(classnames, templates, processor, model, device)
        
        # Load dataset
        hf_dataset = _load_dataset_local_or_hub(hf_id, split, data_dir=data_dir, streaming=True)
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
    parser.add_argument("--data_dir", type=str, default=None)
    
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
            data_dir=args.data_dir
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
            data_dir=args.data_dir
        )
        print(f"\n🎯 Final: Top-1={result['top1_accuracy']:.2f}%")
