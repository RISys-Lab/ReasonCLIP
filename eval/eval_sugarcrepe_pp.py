import argparse
import os
import io
import re
import ast

import torch
import numpy as np
from datasets import load_dataset, get_dataset_config_names
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import open_clip
from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor
import sys


def _infer_model_type(name: str | None) -> str:
    """
    Infer model family from a free-form string by substring match.
    Rule: lowercase then check if it contains "siglip2" -> "siglip2", "siglip" -> "siglip", "clip" -> "clip".
    Defaults to "clip" when unknown.
    """
    if name is None:
        return "clip"
    s = str(name).lower()
    if "siglip2" in s:
        return "siglip2"  # SigLIP2 需要小写文本
    if "siglip" in s:
        return "siglip"
    if "open_clip" in s or "openclip" in s or "::" in s:
        return "open_clip"
    if "longclip" in s:
        return "longclip"
    if "pe-core" in s or s.startswith("pe"):
        return "pe"
    if "clip" in s:
        return "clip"
    return "clip"


def _parse_available_configs_from_err(msg: str) -> list[str]:
    """
    datasets sometimes raises:
      ValueError: Couldn't find cache for <ds> for config 'default'
      Available configs in the cache: [...]
    We parse the list so we can fall back automatically when offline / default config doesn't exist.
    """
    m = re.search(r"Available configs in the cache:\s*(\[[^\]]*\])", msg)
    if not m:
        return []
    try:
        v = ast.literal_eval(m.group(1))
    except Exception:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return []


def _load_model_and_processor(
    model_id: str,
    model_type: str | None,
    processor_name: str | None,
    device: str | None,
):
    """
    Load model + processor once and reuse across subset evaluations.
    Returns: (model, processor, resolved_model_type, resolved_processor_name, resolved_device, use_lowercase)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect model type:
    # - if user passes "auto"/None -> infer from model_id
    # - else -> infer from the provided string (can be "clip", "siglip", "siglip2", or a model name/path)
    mt = "auto" if model_type is None else str(model_type).strip().lower()
    if mt == "auto" or mt == "":
        model_type = _infer_model_type(model_id)
    else:
        model_type = _infer_model_type(model_type)

    # SigLIP2 需要小写文本
    use_lowercase = (model_type.lower() == "siglip2")
    # siglip2 被当作 siglip 处理模型加载
    effective_model_type = "siglip" if model_type.lower() == "siglip2" else model_type.lower()

    if effective_model_type == "open_clip":
        model_name, pretrained = model_id.split("::")
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        resolved_processor_name = "open_clip"
        processor = {
            "image_preprocess": image_preprocess,
            "tokenizer": open_clip.get_tokenizer(model_name),
        }
        print(f"Loaded OpenCLIP model: {model_name} ({pretrained})")
    elif effective_model_type == "longclip":
        longclip_root = os.path.join(os.path.dirname(__file__), "Long-CLIP")
        if longclip_root not in sys.path:
            sys.path.insert(0, longclip_root)
        from model import longclip
        model, image_preprocess = longclip.load(model_id, device=device)
        resolved_processor_name = "longclip"
        processor = {
            "image_preprocess": image_preprocess,
            "tokenizer": longclip.tokenize,
        }
        print(f"Loaded LongCLIP model: {model_id}")
    elif effective_model_type == "pe":
        if os.path.dirname(__file__) not in sys.path:
            sys.path.insert(0, os.path.dirname(__file__))
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        model = pe.CLIP.from_config(model_id, pretrained=True)
        resolved_processor_name = "pe"
        processor = {
            "image_preprocess": transforms.get_image_transform(model.image_size),
            "tokenizer": transforms.get_text_tokenizer(model.context_length),
        }
        print(f"Loaded PE model: {model_id}")
    elif effective_model_type == "clip":
        model = AutoModel.from_pretrained(model_id)
        if processor_name is None:
            resolved_processor_name = model_id
        else:
            resolved_processor_name = processor_name
        processor = AutoProcessor.from_pretrained(resolved_processor_name)
        print(f"Loaded CLIP model: {model_id} and processor: {resolved_processor_name}")
    elif effective_model_type == "siglip":
        model = SiglipModel.from_pretrained(model_id)
        if processor_name is None:
            resolved_processor_name = model_id
        else:
            resolved_processor_name = processor_name
        processor = SiglipProcessor.from_pretrained(resolved_processor_name)
        print(f"Loaded SigLIP model: {model_id} and processor: {resolved_processor_name}")
        if use_lowercase:
            print(f"📝 SigLIP2 检测到: 将对所有文本进行小写转换")
    else:
        raise ValueError("model_type must be one of: open_clip, longclip, pe, clip, siglip, siglip2, auto")

    model.to(device).eval()
    return model, processor, model_type, resolved_processor_name, device, use_lowercase


class SugarCrepePPDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Aman-J/SugarCrepe_pp.

    Expected columns per row:
    - filename: e.g. "xxx.jpg" (image will be loaded from image_dir/filename)
    - caption: positive text 1
    - caption2: positive text 2
    - negative_caption: negative text
    """

    def __init__(self, hf_ds, image_dir: str, max_samples: int | None = None):
        self.ds = hf_ds
        self.image_dir = image_dir
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return len(self.ds)
        return min(self.max_samples, len(self.ds))

    def _load_image(self, filename: str) -> Image.Image:
        path = os.path.join(self.image_dir, filename)
        try:
            return Image.open(path).convert("RGB")
        except FileNotFoundError:
            # Fallback: blank image, but keep the pipeline running
            return Image.new("RGB", (224, 224), color="black")
        except Exception:
            return Image.new("RGB", (224, 224), color="black")

    def __getitem__(self, idx):
        row = self.ds[idx]
        filename = row.get("filename")
        if filename is None:
            raise KeyError("Dataset row missing 'filename'")

        image = self._load_image(str(filename))

        # Three texts: P1, P2, N
        p1 = row.get("caption", "")
        p2 = row.get("caption2", "")
        n = row.get("negative_caption", "")

        # Robustness: sometimes caption2 may be missing/None
        p1 = "" if p1 is None else str(p1)
        p2 = "" if p2 is None else str(p2)
        n = "" if n is None else str(n)

        texts = [p1, p2, n]
        return image, texts, str(filename)


def collate_sugarcrepe_pp(batch, processor, model_type="clip", lowercase=False):
    images, texts_3, filenames = zip(*batch)

    if model_type in ("open_clip", "longclip", "pe"):
        image_tensors = torch.stack([processor["image_preprocess"](img) for img in images], dim=0)
        image_inputs = {"pixel_values": image_tensors}
    else:
        image_inputs = processor(images=list(images), return_tensors="pt")

    # Flatten texts: [B, 3] -> [3B]
    flat_texts = []
    for t3 in texts_3:
        flat_texts.extend(list(t3))
    
    # SigLIP2 需要小写文本
    if lowercase:
        flat_texts = [t.lower() if isinstance(t, str) else t for t in flat_texts]

    if model_type in ("open_clip", "longclip", "pe"):
        text_inputs = {"input_ids": processor["tokenizer"](flat_texts)}
    else:
        proc_name = processor.__class__.__name__.lower()
        text_max_len = 64 if "siglip" in proc_name else 77
        text_inputs = processor(
            text=flat_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_max_len,
        )

    return image_inputs, text_inputs, list(filenames)


def _load_hf_dataset(dataset_name: str, split: str, config_name: str | None = None):
    """
    Load dataset with (optional) config/subset name and split.
    SugarCrepe_pp uses multiple configs (subsets), e.g. replace_attribute, swap_object, ...
    """
    if config_name is not None:
        # In HF datasets, config name is passed as the second positional argument
        try:
            return load_dataset(dataset_name, config_name, split=split)
        except ValueError as e:
            # Sometimes config_name is "default" even though the cached dataset only has
            # real configs like replace_attribute/swap_object (offline mode).
            configs = _parse_available_configs_from_err(str(e))
            if configs and str(config_name).strip().lower() == "default":
                fallback_cfg = configs[0]
                print(
                    f"[WARN] Requested config '{config_name}' is not available in cache for '{dataset_name}'. "
                    f"Falling back to cached config '{fallback_cfg}'. "
                    f"Pass --subset to choose explicitly."
                )
                return load_dataset(dataset_name, fallback_cfg, split=split)
            raise

    # If user didn't specify config, try "default" behavior first.
    try:
        return load_dataset(dataset_name, split=split)
    except ValueError as e:
        # Common when dataset has multiple configs but no 'default', especially in offline mode.
        configs = _parse_available_configs_from_err(str(e))
        if configs:
            fallback_cfg = configs[0]
            print(
                f"[WARN] Dataset '{dataset_name}' has no usable default config in cache. "
                f"Falling back to config '{fallback_cfg}'. "
                f"Pass --subset to choose explicitly."
            )
            return load_dataset(dataset_name, fallback_cfg, split=split)
        raise


def run_sugarcrepe_pp_eval(
    model_id: str,
    model_type: str,
    dataset_name: str,
    split: str,
    image_dir: str,
    batch_size: int,
    device: str | None,
    max_samples: int | None,
    results_dir: str | None,
    processor_name: str | None = None,
    config_name: str | None = None,
    save_json: bool = True,
    save_txt: bool = True,
    skip_if_exists: bool = False,
    model=None,
    processor=None,
    use_lowercase: bool = False,
):
    if results_dir is not None and (save_json or save_txt):
        safe_model = model_id.replace("/", "_")
        safe_ds = dataset_name.replace("/", "_")
        safe_subset = (config_name or "all").replace("/", "_")
        stem = f"sugarcrepe_pp_{model_type}_{safe_model}_{safe_ds}_{safe_subset}_{split}"
        if skip_if_exists and save_txt:
            txt_path = os.path.join(results_dir, f"{stem}.txt")
            if os.path.isfile(txt_path):
                print(f"[SKIP] Results already exist: {txt_path}")
                return {"skipped": True, "txt_path": txt_path}

    # If model/processor are not provided, load them here (single-subset mode).
    # In ALL-SUBSETS mode, caller passes preloaded objects to avoid reloading 5x.
    if model is None or processor is None:
        model, processor, model_type, processor_name, device, use_lowercase = _load_model_and_processor(
            model_id=model_id,
            model_type=model_type,
            processor_name=processor_name,
            device=device,
        )
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Image dir: {image_dir}")
    print(f"Batch size: {batch_size}")
    if max_samples is not None:
        print(f"Max samples: {max_samples}")

    ds = _load_hf_dataset(dataset_name, split, config_name=config_name)
    cfg_msg = f", subset={config_name}" if config_name is not None else ""
    print(f"Loaded dataset rows: {len(ds)}{cfg_msg}")
    print(f"Columns: {list(ds.features.keys()) if hasattr(ds, 'features') else 'unknown'}")

    dataset = SugarCrepePPDataset(ds, image_dir=image_dir, max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_sugarcrepe_pp(b, processor, model_type=model_type, lowercase=use_lowercase),
    )

    # Metrics:
    # Paper definition (SugarCrepe++):
    # - ITT_hit = 1 iff (p(P1|I) > p(N|I)) AND (p(P2|I) > p(N|I)), else 0
    # - TOT_hit = 1 iff (p(P1|P2) > p(N|P2)) AND (p(P2|P1) > p(N|P1)), else 0
    # For embedding models (CLIP/SigLIP), log-likelihood is proportional to cosine similarity,
    # so we implement the same logic with dot product on normalized embeddings.
    total = 0
    itt_correct = 0
    tot_correct = 0

    # Optional: keep running mean of similarities (debug)
    sims_sum_itt = torch.zeros(3, dtype=torch.float64)
    sims_sum_tot = torch.zeros(3, dtype=torch.float64)  # [P1P2, P1N, P2N]

    with torch.no_grad():
        for image_inputs, text_inputs, filenames in tqdm(dataloader, desc="Evaluating"):
            if model_type in ("open_clip", "longclip", "pe"):
                image_features = model.encode_image(image_inputs["pixel_values"].to(device))
                text_features = model.encode_text(text_inputs["input_ids"].to(device))
            else:
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            B = image_features.shape[0]
            # reshape text features back to [B, 3, D]
            text_features = text_features.view(B, 3, -1)

            # ITT: similarity per sample: [B, 3] where 0=P1, 1=P2, 2=N
            sims = torch.einsum("bd,bkd->bk", image_features, text_features)

            # ITT_hit: (sim(I,P1) > sim(I,N)) AND (sim(I,P2) > sim(I,N))
            itt_hit = (sims[:, 0] > sims[:, 2]) & (sims[:, 1] > sims[:, 2])
            itt_correct += itt_hit.sum().item()
            total += B

            sims_sum_itt += sims.double().sum(dim=0).cpu()

            # TOT: text-only similarities
            p1 = text_features[:, 0, :]
            p2 = text_features[:, 1, :]
            n = text_features[:, 2, :]
            sim_p1p2 = (p1 * p2).sum(dim=1)  # [B]
            sim_p1n = (p1 * n).sum(dim=1)    # [B]
            sim_p2n = (p2 * n).sum(dim=1)    # [B]

            # TOT_hit: (sim(P2,P1) > sim(P2,N)) AND (sim(P1,P2) > sim(P1,N))
            tot_hit = (sim_p1p2 > sim_p2n) & (sim_p1p2 > sim_p1n)
            tot_correct += tot_hit.sum().item()

            sims_sum_tot += torch.stack(
                [sim_p1p2.double(), sim_p1n.double(), sim_p2n.double()],
                dim=1,
            ).sum(dim=0).cpu()

    itt_acc = itt_correct / max(1, total)
    tot_acc = tot_correct / max(1, total)

    mean_sims_itt = (sims_sum_itt / max(1, total)).tolist()  # [P1, P2, N] wrt image
    mean_sims_tot = (sims_sum_tot / max(1, total)).tolist()  # [P1P2, P1N, P2N]

    print("\n" + "=" * 70)
    print("SUGARCREPE_PP EVAL")
    print("=" * 70)
    print(f"Total samples: {total}")
    print("\n[ITT] Image+Text negative discrimination")
    print(f"Correct (argmin == negative_caption): {itt_correct}")
    print(f"Accuracy: {itt_acc * 100:.2f}%")
    print(f"Mean similarity [sim(I,P1), sim(I,P2), sim(I,N)]: {[round(x, 6) for x in mean_sims_itt]}")
    print("\n[TOT] Text-only consistency")
    print(f"Correct (both inequalities hold): {tot_correct} / {total}")
    print(f"Accuracy: {tot_acc * 100:.2f}%")
    print(f"Mean similarity [sim(P1,P2), sim(P1,N), sim(P2,N)]: {[round(x, 6) for x in mean_sims_tot]}")
    print("=" * 70)

    result = {
        "model": model_id,
        "model_type": model_type,
        "dataset": dataset_name,
        "subset": config_name,
        "split": split,
        "image_dir": image_dir,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "total": total,
        "ITT": {
            "task": "ITT",
            "metric": "argmin(sim(I,P1), sim(I,P2), sim(I,N)) == N",
            "correct": itt_correct,
            "accuracy": itt_acc * 100.0,
            "mean_similarity_simI_P1_P2_N": mean_sims_itt,
        },
        "TOT": {
            "task": "TOT",
            "metric": "sim(P2,P1) > sim(P2,N) and sim(P1,P2) > sim(P1,N) (both must hold)",
            "correct": tot_correct,
            "denom": total,
            "accuracy": tot_acc * 100.0,
            "mean_similarity_simP1P2_simP1N_simP2N": mean_sims_tot,
        },
        "processor_name": processor_name,
    }

    if results_dir is not None and (save_json or save_txt):
        os.makedirs(results_dir, exist_ok=True)
        safe_model = model_id.replace("/", "_")
        safe_ds = dataset_name.replace("/", "_")
        safe_subset = (config_name or "all").replace("/", "_")
        stem = f"sugarcrepe_pp_{model_type}_{safe_model}_{safe_ds}_{safe_subset}_{split}"

        if save_json:
            out_path = os.path.join(results_dir, f"{stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                import json

                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved results to: {out_path}")

        if save_txt:
            txt_path = os.path.join(results_dir, f"{stem}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"ITT\t{result['ITT']['accuracy']:.4f}\n")
                f.write(f"TOT\t{result['TOT']['accuracy']:.4f}\n")
            print(f"Saved summary to: {txt_path}")

    return result


def run_sugarcrepe_pp_eval_by_subsets(
    model_id: str,
    model_type: str,
    dataset_name: str,
    split: str,
    image_dir: str,
    batch_size: int,
    device: str | None,
    max_samples: int | None,
    results_dir: str | None,
    processor_name: str | None = None,
    skip_if_exists: bool = False,
):
    """
    Evaluate ITT/TOT on each dataset subset (config) and write a single summary txt.
    """
    try:
        subsets = get_dataset_config_names(dataset_name)
    except Exception:
        subsets = []
    # Offline/cached edge case: get_dataset_config_names may return ["default"] even when
    # the cached dataset only has real configs (replace_attribute, swap_object, ...).
    if not subsets or (len(subsets) == 1 and str(subsets[0]).strip().lower() == "default"):
        cached_configs: list[str] = []
        try:
            # Force a cache lookup that triggers the helpful "Available configs in the cache" message.
            load_dataset(dataset_name, "default", split=split)
        except ValueError as e:
            cached_configs = _parse_available_configs_from_err(str(e))
        except Exception:
            cached_configs = []

        if cached_configs:
            subsets = cached_configs
        else:
            # Last resort: run a single pass and let _load_hf_dataset auto-pick a cached config.
            subsets = [None]

    # Skip if the consolidated output already exists.
    if results_dir is not None and skip_if_exists:
        safe_model = model_id.replace("/", "_")
        safe_ds = dataset_name.replace("/", "_")
        stem = f"sugarcrepe_pp_{model_type}_{safe_model}_{safe_ds}_{split}_ALL_SUBSETS"
        txt_path = os.path.join(results_dir, f"{stem}.txt")
        if os.path.isfile(txt_path):
            print(f"[SKIP] Results already exist: {txt_path}")
            return []

    # Load model/processor ONCE and reuse across subsets.
    model, processor, resolved_model_type, resolved_processor_name, device, use_lowercase = _load_model_and_processor(
        model_id=model_id,
        model_type=model_type,
        processor_name=processor_name,
        device=device,
    )

    all_results = []
    for cfg in subsets:
        print("\n" + "=" * 80)
        print(f"Evaluating subset: {cfg}")
        print("=" * 80)
        res = run_sugarcrepe_pp_eval(
            model_id=model_id,
            model_type=resolved_model_type,
            dataset_name=dataset_name,
            split=split,
            image_dir=image_dir,
            batch_size=batch_size,
            device=device,
            max_samples=max_samples,
            # Do NOT save per-subset files; we'll save one consolidated txt at the end.
            results_dir=None,
            processor_name=resolved_processor_name,
            config_name=cfg,
            save_json=False,
            save_txt=False,
            model=model,
            processor=processor,
            use_lowercase=use_lowercase,
        )
        all_results.append(res)

    # One combined txt:
    # 1) ITT/TOT overall micro-average
    # 2) per-subset ITT/TOT (10 lines for 5 subsets)
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        safe_model = model_id.replace("/", "_")
        safe_ds = dataset_name.replace("/", "_")
        stem = f"sugarcrepe_pp_{model_type}_{safe_model}_{safe_ds}_{split}_ALL_SUBSETS"
        txt_path = os.path.join(results_dir, f"{stem}.txt")

        # Macro-average over subsets (equal weight per subset), as requested.
        # Note: result["ITT"]["accuracy"] / ["TOT"]["accuracy"] are already in PERCENT units.
        n = max(1, len(all_results))
        itt_avg_pct = sum(float(r["ITT"]["accuracy"]) for r in all_results) / n
        tot_avg_pct = sum(float(r["TOT"]["accuracy"]) for r in all_results) / n

        with open(txt_path, "w", encoding="utf-8") as f:
            # First: two task averages
            f.write(f"ITT\t{itt_avg_pct:.4f}\n")
            f.write(f"TOT\t{tot_avg_pct:.4f}\n")
            f.write("\n")
            # Then: per-subset results (10 lines)
            for res in all_results:
                subset = res.get("subset") or "default"
                f.write(f"{subset}\tITT\t{res['ITT']['accuracy']:.4f}\n")
                f.write(f"{subset}\tTOT\t{res['TOT']['accuracy']:.4f}\n")
        print(f"\nSaved ALL-SUBSETS summary to: {txt_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="SugarCrepe_pp negative caption discrimination eval")

    parser.add_argument("--model_path", type=str, default="/home/muzammal/.cache/huggingface/hub/models--fesvhtr--clip336-r-s2-run1218-505/snapshots/f2b8cf27d26196ce98d8109df1986f34b2b4163b", help="HF model id or local path")
    # Free-form model name/type string. We infer by substring match on lowercase:
    # contains "siglip" -> SigLIP, else contains "clip" -> CLIP, else default CLIP.
    # Use "auto" to force inferring from --model_path.
    parser.add_argument("--model_name", type=str, default="auto", help="Model type/name (auto-detected by substring match)")

    parser.add_argument("--dataset_name", type=str, default="Aman-J/SugarCrepe_pp", help="HF dataset id")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (SugarCrepe_pp typically uses 'train')")

    parser.add_argument("--image_dir", type=str, default="/home/muzammal/Projects/CLIP-R/data/val2017", help="Local folder containing images, joined with filename")

    parser.add_argument("--batch_size", type=int, default=384, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:1", help="cuda:0/cpu; default auto")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional HF dataset config/subset name (e.g. replace_attribute). "
        "If omitted, we run ALL subsets when available; if offline and configs can't be fetched, we'll fall back to a cached config.",
    )

    parser.add_argument(
        "--processor_name",
        type=str,
        default=None,
        help="Optional processor name/path. If not set, use model-matched default (usually same as --model_path).",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/muzammal/Projects/CLIP-R/eval/results/sugarcrepe_pp",
        help="Directory to save json result (set to empty to disable saving)",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip evaluation when output txt already exists",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_dir = args.results_dir
    if results_dir is not None and str(results_dir).strip() == "":
        results_dir = None

    if args.subset is not None and str(args.subset).strip() != "":
        run_sugarcrepe_pp_eval(
            model_id=args.model_path,
            model_type=args.model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            device=args.device,
            max_samples=args.max_samples,
            results_dir=results_dir,
            processor_name=args.processor_name,
            config_name=args.subset,
            save_json=True,
            save_txt=True,
            skip_if_exists=args.skip_if_exists,
        )
    else:
        run_sugarcrepe_pp_eval_by_subsets(
            model_id=args.model_path,
            model_type=args.model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            device=args.device,
            max_samples=args.max_samples,
            results_dir=results_dir,
            processor_name=args.processor_name,
            skip_if_exists=args.skip_if_exists,
        )

