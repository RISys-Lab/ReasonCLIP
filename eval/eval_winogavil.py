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
    skip_if_exists=False
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if processor_path is None:
        processor_path = model_path

    # --- 1. Setup Results Directory & Check Skip Logic ---
    if results_dir is None:
        results_dir = os.path.join(SCRIPT_DIR, "results")
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
    print("=" * 80)

    # --- 2. Load Model & Processor ---
    model_type = _infer_model_type(model_path)
    is_siglip = model_type in ("siglip", "siglip2")
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
        for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
            
            # Extract Data
            cue_word = sample.get("cue")
            candidates_text = sample.get("candidates")
            associations_text = sample.get("associations")
            k = sample.get("num_associations") 
            candidate_images = sample.get("candidate_images")
            
            if not candidate_images or not cue_word:
                continue

            # --- Text Processing (Updated Logic) ---
            if use_lowercase:
                cue_word = cue_word.lower()
            
            prompt = template.format(cue_word)
            
            # STRICT ALIGNMENT with your ImageNet script:
            if is_siglip:
                text_inputs = processor(
                    text=[prompt],
                    return_tensors="pt",
                    padding="max_length", # Specific to SigLIP script logic
                    truncation=True,
                    max_length=64         # Specific to SigLIP script logic
                )
            else:
                text_inputs = processor(
                    text=[prompt],
                    return_tensors="pt",
                    padding=True,         # Default for CLIP
                    truncation=True
                )
            
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            # Image Processing
            image_inputs = processor(images=candidate_images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            # Inference
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Similarity
            logits = text_features @ image_features.T
            scores = logits.squeeze(0) 

            # Top-K Selection
            _, topk_indices = scores.topk(k)
            pred_indices = topk_indices.cpu().tolist()

            # Map GT text to indices
            target_indices = []
            for assoc in associations_text:
                if assoc in candidates_text:
                    target_indices.append(candidates_text.index(assoc))
            
            # Jaccard Calculation
            jaccard = compute_jaccard(pred_indices, target_indices)
            
            total_jaccard += jaccard
            total_samples += 1

            # Track by difficulty
            num_cand = len(candidate_images)
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
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--template", type=str, default="a {}", help="Prompt template")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--max_samples", type=int, default=None)
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
        skip_if_exists=args.skip_if_exists
    )