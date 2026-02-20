import argparse
from collections import defaultdict
import os
from datetime import datetime
from contextlib import nullcontext

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import open_clip
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor
from tqdm import tqdm


def _infer_model_type(name: str | None) -> str:
    if name is None:
        return "clip"
    s = str(name).lower()
    if "siglip2" in s:
        return "siglip2"  # SigLIP2 需要小写文本
    if "siglip" in s:
        return "siglip"
    if "open_clip" in s or "openclip" in s or "::" in s:
        return "open_clip"
    if "metaclip" in s:
        return "metaclip"
    if "clip" in s:
        return "clip"
    return "clip"


def _get_text_max_len(processor) -> int:
    proc_name = processor.__class__.__name__.lower()
    return 64 if "siglip" in proc_name else 77


def _ensure_rgb_images(images):
    rgb_images = []
    for img in images:
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb_images.append(img)
            continue

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            rgb_images.append(img)
            continue

        rgb_images.append(img)
    return rgb_images


def _encode_image_features(model, processor, images, device, autocast_ctx):
    images = _ensure_rgb_images(images)
    if isinstance(processor, dict):
        image_tensors = torch.stack([processor["image_preprocess"](img) for img in images], dim=0).to(device)
        with autocast_ctx:
            image_features = model.encode_image(image_tensors)
    else:
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        with autocast_ctx:
            image_features = model.get_image_features(**image_inputs)
    return image_features / image_features.norm(dim=-1, keepdim=True)


def _encode_text_features(model, processor, text, device, autocast_ctx, lowercase=False):
    # SigLIP2 需要小写文本
    if lowercase:
        text = [t.lower() if isinstance(t, str) else t for t in text]
    if isinstance(processor, dict):
        text_tokens = processor["tokenizer"](text).to(device)
        with autocast_ctx:
            text_features = model.encode_text(text_tokens)
    else:
        text_max_len = _get_text_max_len(processor)
        text_inputs = processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_max_len,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with autocast_ctx:
            text_features = model.get_text_features(**text_inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def compute_similarity(model, processor, images, text, device, autocast_ctx, lowercase=False):
    image_features = _encode_image_features(model, processor, images, device, autocast_ctx)
    text_features = _encode_text_features(model, processor, text, device, autocast_ctx, lowercase=lowercase)
    return image_features @ text_features.t()


def get_image_to_text_score(model, processor, images, text, device, autocast_ctx, return_tot=False, lowercase=False):
    similarity_scores = compute_similarity(model, processor, images, text, device, autocast_ctx, lowercase=lowercase)
    if not return_tot:
        return int(similarity_scores.argmax() == 0), similarity_scores

    i0_c0 = similarity_scores[0, 0].item()
    i0_c1 = similarity_scores[0, 1].item()
    i0_c2 = similarity_scores[0, 2].item()
    image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

    text_features = _encode_text_features(model, processor, text, device, autocast_ctx, lowercase=lowercase)
    text_similarity_scores = text_features @ text_features.t()
    c0_c1 = text_similarity_scores[0, 1].item()
    c0_c2 = text_similarity_scores[0, 2].item()
    c1_c2 = text_similarity_scores[1, 2].item()
    text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

    return int(image_correct), similarity_scores, int(text_correct), [c0_c1, c0_c2, c1_c2]


def eval_whatsup(model, processor, device, autocast_ctx, dataset_path=None, lowercase=False):
    if dataset_path:
        dataset = load_dataset(dataset_path, trust_remote_code=True, split="test")
    else:
        dataset = load_dataset("Mayfull/whats_up_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in tqdm(dataset, desc="Evaluating WhatsUp"):
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device, autocast_ctx, lowercase=lowercase)
        if sample["original_file_name"].startswith("coco"):
            name = "coco"
        elif sample["original_file_name"].startswith("vg"):
            name = "vg"
        else:
            name = "whatsup"
        result[name].append(score)
    average_result = {k: 100 * round(sum(v) / len(v), 5) for k, v in result.items()}
    average_result["total"] = round(sum(average_result.values()) / len(average_result), 3)
    return average_result


def eval_valse(model, processor, device, autocast_ctx, dataset_path=None, lowercase=False):
    if dataset_path:
        dataset = load_dataset(dataset_path, trust_remote_code=True, split="test")
    else:
        dataset = load_dataset("Mayfull/valse_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in tqdm(dataset, desc="Evaluating VALSE"):
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device, autocast_ctx, lowercase=lowercase)
        result[sample["linguistic_phenomena"]].append(score)
        result["total"].append(score)
    average_result = {k: 100 * round(sum(v) / len(v), 5) for k, v in result.items()}
    return average_result


def eval_crepe(model, processor, device, autocast_ctx, dataset_path=None, lowercase=False):
    if dataset_path:
        dataset = load_dataset(dataset_path, trust_remote_code=True, split="test")
    else:
        dataset = load_dataset("Mayfull/crepe_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in tqdm(dataset, desc="Evaluating CREPE"):
        images = [
            i.crop(
                (
                    sample["x"],
                    sample["y"],
                    sample["x"] + sample["width"],
                    sample["y"] + sample["height"],
                )
            )
            for i in sample["images"]
        ]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device, autocast_ctx, lowercase=lowercase)
        result[sample["original_file_name"]].append(score)
    average_result = {k: 100 * round(sum(v) / len(v), 5) for k, v in result.items()}
    average_result["total"] = round(sum(average_result.values()) / len(average_result), 3)
    return average_result


def eval_sugarcrepe(model, processor, device, autocast_ctx, dataset_path=None, lowercase=False):
    if dataset_path:
        dataset = load_dataset(dataset_path, trust_remote_code=True, split="test")
    else:
        dataset = load_dataset("Mayfull/sugarcrepe_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in tqdm(dataset, desc="Evaluating SugarCrepe"):
        images = sample["images"]
        text = [*sample["positive_caption"], *sample["negative_caption"]]
        score, _ = get_image_to_text_score(model, processor, images, text, device, autocast_ctx, lowercase=lowercase)
        result[sample["original_file_name"]].append(score)
    average_result = {k: 100 * round(sum(v) / len(v), 5) for k, v in result.items()}
    average_result["total"] = round(sum(average_result.values()) / len(average_result), 3)
    return average_result


def eval_sugarcrepe_pp(model, processor, device, autocast_ctx, dataset_path=None, lowercase=False):
    if dataset_path:
        dataset = load_dataset(dataset_path, trust_remote_code=True, split="test")
    else:
        dataset = load_dataset("Mayfull/sugarcrepepp_vlms", trust_remote_code=True, split="test")
    result = defaultdict(list)
    for sample in tqdm(dataset, desc="Evaluating SugarCrepe++"):
        images = sample["images"]
        text = [*sample["positive_caption_1"], *sample["positive_caption_2"], *sample["negative_caption"]]
        itt_score, _, tot_score, _ = get_image_to_text_score(model, processor, images, text, device, autocast_ctx, return_tot=True, lowercase=lowercase)
        result[sample["original_file_name"] + "_itt"].append(itt_score)
        result[sample["original_file_name"] + "_tot"].append(tot_score)
    average_result = {k: 100 * round(sum(v) / len(v), 5) for k, v in result.items()}
    average_result["total_itt"] = round(
        sum(v for k, v in average_result.items() if k.endswith("itt")) / sum(
            k.endswith("itt") for k in average_result
        ), 3
    )
    average_result["total_tot"] = round(
        sum(v for k, v in average_result.items() if k.endswith("tot")) / sum(
            k.endswith("tot") for k in average_result
        ), 3
    )
    return average_result


def parse_args():
    parser = argparse.ArgumentParser(description="Compositional reasoning evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--processor_path", type=str, default=None, help="Processor path (default: same as model_path)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval/results/compositional_results",
        help="Results directory (default: eval/results/compositional_results)",
    )
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 autocast (use fp32)")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip evaluation when output txt already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor_path = args.processor_path or args.model_path
    use_bf16 = not args.no_bf16
    model_type = _infer_model_type(args.model_path)
    
    # SigLIP2 需要小写文本
    use_lowercase = (model_type == "siglip2")
    # siglip2 被当作 siglip 处理模型加载
    effective_model_type = "siglip" if model_type == "siglip2" else model_type

    os.makedirs(args.results_dir, exist_ok=True)
    model_name = args.model_path.replace("/", "_")
    result_file = os.path.join(args.results_dir, f"compositional_{model_name}.txt")
    if args.skip_if_exists and os.path.isfile(result_file):
        print(f"[SKIP] Results already exist: {result_file}")
        return
    
    print("=" * 80)
    print("Compositional Reasoning Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Model type: {model_type.upper()}")
    print(f"Processor: {processor_path}")
    print(f"Device: {device}")
    print(f"BF16: {'on' if use_bf16 else 'off'}")
    if use_lowercase:
        print(f"📝 SigLIP2 检测到: 将对所有文本进行小写转换")
    print("=" * 80)
    
    # Load model and processor
    print("Loading model and processor...")
    torch_dtype = torch.bfloat16 if use_bf16 and device != "cpu" else None
    if effective_model_type == "siglip":
        model = SiglipModel.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device).eval()
        processor = SiglipProcessor.from_pretrained(processor_path)
    elif effective_model_type == "open_clip":
        model_name, pretrained = args.model_path.split("::")
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        processor = {
            "image_preprocess": image_preprocess,
            "tokenizer": open_clip.get_tokenizer(model_name),
        }
        model = model.to(device).eval()
    elif effective_model_type in ("clip", "metaclip"):
        model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(processor_path)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Must contain 'open_clip', 'metaclip', 'clip', 'siglip', or 'siglip2'."
        )
    
    # Setup autocast context
    if use_bf16 and device != "cpu":
        autocast_ctx = torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()
    
    # Run all evaluations
    print("\nRunning evaluations...")
    results = {
        "WhatsUp": eval_whatsup(model, processor, device, autocast_ctx, lowercase=use_lowercase),
        "VALSE": eval_valse(model, processor, device, autocast_ctx, lowercase=use_lowercase),
        "CREPE": eval_crepe(model, processor, device, autocast_ctx, lowercase=use_lowercase),
        "SugarCrepe": eval_sugarcrepe(model, processor, device, autocast_ctx, lowercase=use_lowercase),
        "SugarCrepe++": eval_sugarcrepe_pp(model, processor, device, autocast_ctx, lowercase=use_lowercase),
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for name, metrics in results.items():
        print(f"\n{name}:")
        if name == "SugarCrepe++":
            print(f"  ITT Total: {metrics['total_itt']:.2f}%")
            print(f"  TOT Total: {metrics['total_tot']:.2f}%")
        else:
            print(f"  Total: {metrics.get('total', 0):.2f}%")
    print("=" * 80)
    
    # Save results
    
    with open(result_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Compositional Reasoning Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Processor: {processor_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write("-" * 40 + "\n")
            if name == "SugarCrepe++":
                f.write(f"  ITT Total: {metrics['total_itt']:.2f}%\n")
                f.write(f"  TOT Total: {metrics['total_tot']:.2f}%\n")
                # Write subcategories
                for k, v in metrics.items():
                    if k not in ["total_itt", "total_tot"]:
                        f.write(f"    {k}: {v:.2f}%\n")
            else:
                f.write(f"  Total: {metrics.get('total', 0):.2f}%\n")
                # Write subcategories
                for k, v in metrics.items():
                    if k != "total":
                        f.write(f"    {k}: {v:.2f}%\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Summary:\n")
        f.write("-" * 40 + "\n")
        for name, metrics in results.items():
            if name == "SugarCrepe++":
                f.write(f"{name:20s} ITT: {metrics['total_itt']:6.2f}%  TOT: {metrics['total_tot']:6.2f}%\n")
            else:
                f.write(f"{name:20s} {metrics.get('total', 0):6.2f}%\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Results saved to: {result_file}")


if __name__ == "__main__":
    main()
