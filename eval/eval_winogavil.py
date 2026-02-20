"""
Zero-shot WinoGAViL evaluation (Strict Alignment Version)
Supports: nlphuji/winogavil
Includes: Output directory, Skip logic, and Specific SigLIP text processing
"""
import torch
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from contextlib import nullcontext
import open_clip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _infer_model_type(name: str | None) -> str:
    """
    Infer model family from a free-form string by substring match.
    Same logic as reference script.
    """
    if name is None:
        return "clip"
    s = str(name).lower()
    if "siglip2" in s:
        return "siglip2"
    if "siglip" in s:
        return "siglip"
    if "open_clip" in s or "openclip" in s or "::" in s:
        return "open_clip"
    if "metaclip" in s:
        return "metaclip"
    if "clip" in s:
        return "clip"
    return "clip"

def compute_jaccard(pred_indices, target_indices):
    """Compute Jaccard Index for a single sample."""
    set_pred = set(pred_indices)
    set_target = set(target_indices)
    
    intersection = len(set_pred & set_target)
    union = len(set_pred | set_target)
    
    if union == 0:
        return 0.0
    return intersection / union

def run_winogavil_evaluation(
    model_path,
    processor_path=None,
    dataset_name="nlphuji/winogavil",
    split="test",
    device=None,
    results_dir=None,
    use_bf16=True,
    max_samples=None,
    template="a {}",
    skip_if_exists=False,
    batch_size=1
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if processor_path is None:
        processor_path = model_path

    # --- 1. Setup Results Directory & Check Skip Logic ---
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results", "winogavil")
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct filename based on model and dataset
    model_name_safe = model_path.replace("/", "_")
    result_filename = f"winogavil_{model_name_safe}.txt"
    result_file = os.path.join(results_dir, result_filename)

    # CHECK: Skip if exists
    if skip_if_exists and os.path.isfile(result_file):
        print("=" * 80)
        print(f"⏩ [SKIP] Results already exist: {result_file}")
        print("=" * 80)
        return

    print("=" * 80)
    print("🚀 Running WinoGAViL Zero-Shot Evaluation (Strict Mode)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} [{split}]")
    print(f"Output File: {result_file}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print("=" * 80)

    # --- 2. Load Model & Processor ---
    model_type = _infer_model_type(model_path)
    is_siglip = model_type in ("siglip", "siglip2")
    is_open_clip = model_type == "open_clip"
    use_lowercase = (model_type == "siglip2")
    
    if use_lowercase:
        print(f"📝 SigLIP2 Detected: Forcing lowercase on all text cues.")

    if device != "cpu":
        if is_siglip:
            torch_dtype = torch.bfloat16 if use_bf16 else None
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = None

    print(f"Loading model ({torch_dtype})...")
    try:
        if is_open_clip:
            model_name, pretrained = model_path.split("::")
            model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            processor = {
                "image_preprocess": image_preprocess,
                "tokenizer": open_clip.get_tokenizer(model_name),
            }
        else:
            model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Error loading model/processor: {e}")
        return

    model.to(device).eval()

    # --- 3. Load Dataset ---
    print(f"Loading dataset {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    if max_samples:
        dataset = dataset.select(range(max_samples))
        print(f"⚠️ Limited to first {max_samples} samples.")

    # --- 4. Setup Context ---
    if device != "cpu":
        if is_siglip and use_bf16:
            autocast_ctx = torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.autocast(device_type=device.split(":")[0], dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # --- 5. Evaluation Loop ---
    total_jaccard = 0.0
    total_samples = 0
    metrics_by_candidates = {} # {5: [], 6: [], 10: [], 12: []}

    print("\n🚀 Starting evaluation...")
    
    with torch.no_grad(), autocast_ctx:
        batch = []
        for _, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
            batch.append(sample)
            if len(batch) < batch_size:
                continue

            # Process batch
            cues = []
            candidates_texts = []
            associations_texts = []
            ks = []
            candidate_images_list = []
            offsets = []
            total_imgs = 0

            for s in batch:
                cue_word = s.get("cue")
                candidate_images = s.get("candidate_images")
                candidates_text = s.get("candidates")
                associations_text = s.get("associations")
                k = s.get("num_associations")

                if not candidate_images or not cue_word:
                    continue

                if use_lowercase:
                    cue_word = cue_word.lower()

                cues.append(template.format(cue_word))
                candidates_texts.append(candidates_text)
                associations_texts.append(associations_text)
                ks.append(k)
                candidate_images_list.extend(candidate_images)
                offsets.append((total_imgs, total_imgs + len(candidate_images)))
                total_imgs += len(candidate_images)

            if len(cues) == 0:
                batch = []
                continue

            if is_open_clip:
                text_inputs = {"input_ids": processor["tokenizer"](cues).to(device)}
            elif is_siglip:
                text_inputs = processor(
                    text=cues,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                )
            else:
                text_inputs = processor(
                    text=cues,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            if is_open_clip:
                image_tensors = torch.stack([processor["image_preprocess"](img) for img in candidate_images_list], dim=0).to(device)
                image_features = model.encode_image(image_tensors)
                text_features = model.encode_text(text_inputs["input_ids"])
            else:
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                image_inputs = processor(images=candidate_images_list, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for i in range(len(cues)):
                start, end = offsets[i]
                image_feats_i = image_features[start:end]
                text_feat_i = text_features[i : i + 1]

                logits = text_feat_i @ image_feats_i.T
                scores = logits.squeeze(0)

                k = ks[i]
                if k is None:
                    k = len(associations_texts[i]) if associations_texts[i] else 0

                _, topk_indices = scores.topk(k)
                pred_indices = topk_indices.cpu().tolist()

                target_indices = []
                for assoc in associations_texts[i]:
                    if assoc in candidates_texts[i]:
                        target_indices.append(candidates_texts[i].index(assoc))

                jaccard = compute_jaccard(pred_indices, target_indices)

                total_jaccard += jaccard
                total_samples += 1

                num_cand = end - start
                if num_cand not in metrics_by_candidates:
                    metrics_by_candidates[num_cand] = []
                metrics_by_candidates[num_cand].append(jaccard)

            batch = []

        if len(batch) > 0:
            # Process remaining samples
            cues = []
            candidates_texts = []
            associations_texts = []
            ks = []
            candidate_images_list = []
            offsets = []
            total_imgs = 0

            for s in batch:
                cue_word = s.get("cue")
                candidate_images = s.get("candidate_images")
                candidates_text = s.get("candidates")
                associations_text = s.get("associations")
                k = s.get("num_associations")

                if not candidate_images or not cue_word:
                    continue

                if use_lowercase:
                    cue_word = cue_word.lower()

                cues.append(template.format(cue_word))
                candidates_texts.append(candidates_text)
                associations_texts.append(associations_text)
                ks.append(k)
                candidate_images_list.extend(candidate_images)
                offsets.append((total_imgs, total_imgs + len(candidate_images)))
                total_imgs += len(candidate_images)

            if len(cues) > 0:
                if is_open_clip:
                    text_inputs = {"input_ids": processor["tokenizer"](cues).to(device)}
                elif is_siglip:
                    text_inputs = processor(
                        text=cues,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=64
                    )
                else:
                    text_inputs = processor(
                        text=cues,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                if is_open_clip:
                    image_tensors = torch.stack([processor["image_preprocess"](img) for img in candidate_images_list], dim=0).to(device)
                    image_features = model.encode_image(image_tensors)
                    text_features = model.encode_text(text_inputs["input_ids"])
                else:
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    image_inputs = processor(images=candidate_images_list, return_tensors="pt")
                    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                    image_features = model.get_image_features(**image_inputs)
                    text_features = model.get_text_features(**text_inputs)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                for i in range(len(cues)):
                    start, end = offsets[i]
                    image_feats_i = image_features[start:end]
                    text_feat_i = text_features[i : i + 1]

                    logits = text_feat_i @ image_feats_i.T
                    scores = logits.squeeze(0)

                    k = ks[i]
                    if k is None:
                        k = len(associations_texts[i]) if associations_texts[i] else 0

                    _, topk_indices = scores.topk(k)
                    pred_indices = topk_indices.cpu().tolist()

                    target_indices = []
                    for assoc in associations_texts[i]:
                        if assoc in candidates_texts[i]:
                            target_indices.append(candidates_texts[i].index(assoc))

                    jaccard = compute_jaccard(pred_indices, target_indices)

                    total_jaccard += jaccard
                    total_samples += 1

                    num_cand = end - start
                    if num_cand not in metrics_by_candidates:
                        metrics_by_candidates[num_cand] = []
                    metrics_by_candidates[num_cand].append(jaccard)

    # --- 6. Final Results Calculation ---
    avg_jaccard = (total_jaccard / total_samples) * 100 if total_samples > 0 else 0.0
    
    print("\n" + "=" * 80)
    print(f"📊 Results: {model_path}")
    print("=" * 80)
    print(f"Overall Jaccard Index: {avg_jaccard:.2f}%")
    
    # Save Results
    with open(result_file, "w") as f:
        f.write(f"WinoGAViL Evaluation Results\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Template: {template}\n")
        f.write(f"Overall Jaccard: {avg_jaccard:.2f}%\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write("-" * 30 + "\n")
        
        print("Breakdown by Difficulty:")
        for n_cand in sorted(metrics_by_candidates.keys()):
            scores = metrics_by_candidates[n_cand]
            avg = sum(scores) / len(scores) * 100
            line = f"{n_cand} Candidates: {avg:.2f}% (n={len(scores)})"
            print(f"  {line}")
            f.write(f"{line}\n")
            
    print("=" * 80)
    print(f"💾 Results saved to: {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WinoGAViL Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="nlphuji/winogavil")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save results (default: eval/results/winogavil)",
    )
    parser.add_argument("--template", type=str, default="a {}", help="Prompt template")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if output file exists")
    
    args = parser.parse_args()
    
    run_winogavil_evaluation(
        model_path=args.model_path,
        processor_path=args.processor_path,
        dataset_name=args.dataset,
        results_dir=args.results_dir,
        template=args.template,
        use_bf16=not args.no_bf16,
        max_samples=args.max_samples,
        skip_if_exists=args.skip_if_exists,
        batch_size=args.batch_size
    )
