#!/usr/bin/env python3
#
# API Key: export GEMINI_API_KEY="*"
#
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Tuple, Optional

import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA_BY_VERSION = {
    "v1": "/home/localadmin/bz/RCLIP/rclip_5k_v1_gpt_new.jsonl",
    "v2": "/home/localadmin/bz/RCLIP/rclip_5k_v2_gpt_new.jsonl",
    "v3": "/home/localadmin/bz/RCLIP/rclip_5k_v3_gpt_new.jsonl",
    "v2_gpt5": "/home/localadmin/bz/RCLIP/rclip_5k_v2_gpt5_new_v2.jsonl",
    "v3_gpt5": "/home/localadmin/bz/RCLIP/rclip_5k_v3_gpt5_new_v2.jsonl",
}


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Each line must be a JSON object.")
                yield obj
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Dataset parsing
# -----------------------------
def extract_sets(sample: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Expect sample format:
      {
        "id": ...,
        "image_path": ...,
        "sets": [
          {"tag": "...", "gt": "...", "neg": ["...","...","...","..."]},
          ...
        ]
      }
    """
    sid = str(sample.get("id", ""))
    img_path = sample.get("image_path", "")
    sets = sample.get("sets", [])
    if not isinstance(img_path, str) or not img_path:
        raise ValueError("missing image_path")
    if not isinstance(sets, list) or len(sets) == 0:
        raise ValueError("missing sets")
    return sid, img_path, sets


def validate_set_item(it: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    tag = it.get("tag", "")
    gt = it.get("gt", "")
    neg = it.get("neg", [])
    if not isinstance(tag, str) or not tag:
        raise ValueError("bad tag")
    if not isinstance(gt, str) or not gt.strip():
        raise ValueError("bad gt")
    if not isinstance(neg, list) or len(neg) != 4 or any((not isinstance(x, str) or not x.strip()) for x in neg):
        raise ValueError("bad neg")
    return tag, gt.strip(), [x.strip() for x in neg]


# -----------------------------
# Eval core
# -----------------------------
def _infer_model_type(name: Optional[str]) -> str:
    """
    Infer model family from string content.
    """
    if name is None:
        return "clip"
    s = str(name).lower()
    if "siglip2" in s:
        return "siglip2"
    if "siglip" in s:
        return "siglip"
    if "open_clip" in s or "openclip" in s:
        return "open_clip"
    if "longclip" in s:
        return "longclip"
    if "pe-core" in s or s.startswith("pe"):
        return "pe"
    if "metaclip" in s:
        return "metaclip"
    if "clip" in s:
        return "clip"
    return "clip"


def _maybe_lower_texts(texts: List[str], lowercase: bool) -> List[str]:
    if not lowercase:
        return texts
    return [t.lower() for t in texts]


def _to_device(x: Any, device: torch.device) -> Any:
    # HuggingFace BatchEncoding/BatchFeature support .to(device).
    if hasattr(x, "to"):
        try:
            return x.to(device)
        except Exception:
            pass
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        moved = [_to_device(v, device) for v in x]
        return type(x)(moved) if isinstance(x, tuple) else moved
    return x


def _get_core_model(model: Any) -> Any:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_model_and_processor(
    model_path: str,
    processor_path: Optional[str],
    model_type: str,
    device: torch.device,
) -> Dict[str, Any]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mt = "auto" if model_type is None else str(model_type).strip().lower()
    if mt in ("", "auto"):
        model_type = _infer_model_type(model_path)
    else:
        model_type = _infer_model_type(model_type)

    use_lowercase = (model_type == "siglip2")
    effective_model_type = "siglip" if model_type == "siglip2" else model_type

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if effective_model_type in ("clip", "metaclip", "siglip"):
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype)
        proc_id = processor_path or model_path
        processor = AutoProcessor.from_pretrained(proc_id)
    elif effective_model_type == "open_clip":
        if "::" not in model_path:
            raise ValueError(
                "open_clip model_path must be in format 'model_name::pretrained_tag', "
                "e.g. 'ViT-B-32::laion2b_s34b_b79k'"
            )
        model_name, pretrained = model_path.split("::", 1)
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        processor = {
            "image_preprocess": image_preprocess,
            "tokenizer": open_clip.get_tokenizer(model_name),
        }
    elif effective_model_type == "longclip":
        longclip_root = os.path.join(script_dir, "Long-CLIP")
        if longclip_root not in sys.path:
            sys.path.insert(0, longclip_root)
        from model import longclip
        model, image_preprocess = longclip.load(model_path, device=str(device))
        processor = {
            "image_preprocess": image_preprocess,
            "tokenizer": longclip.tokenize,
        }
    elif effective_model_type == "pe":
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        model = pe.CLIP.from_config(model_path, pretrained=True)
        processor = {
            "image_preprocess": transforms.get_image_transform(model.image_size),
            "tokenizer": transforms.get_text_tokenizer(model.context_length),
        }
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Expected one of "
            "clip/siglip/siglip2/metaclip/open_clip/longclip/pe."
        )

    return {
        "model": model,
        "processor": processor,
        "model_type": model_type,
        "effective_model_type": effective_model_type,
        "use_lowercase": use_lowercase,
        "text_max_len": 64 if effective_model_type == "siglip" else 77,
    }


@torch.inference_mode()
def score_one_set(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    texts: List[str],
    effective_model_type: str = "clip",
    text_max_len: int = 77,
    use_lowercase: bool = False,
) -> torch.Tensor:
    """
    Returns logits_per_image: shape (1, len(texts))
    """
    texts = _maybe_lower_texts(texts, use_lowercase)
    core_model = _get_core_model(model)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16

    if effective_model_type in ("open_clip", "longclip", "pe"):
        image_tensor = processor["image_preprocess"](image).unsqueeze(0).to(device)
        text_tokens = _to_device(processor["tokenizer"](texts), device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.encode_image(image_tensor)
            text_features = core_model.encode_text(text_tokens)
    else:
        image_inputs = processor(images=[image], return_tensors="pt")
        image_inputs = _to_device(image_inputs, device)
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_len,
        )
        text_inputs = _to_device(text_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.get_image_features(**image_inputs)
            text_features = core_model.get_text_features(**text_inputs)

    image_features = image_features.float()
    text_features = text_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = image_features @ text_features.T
    if hasattr(core_model, "logit_scale"):
        scores = core_model.logit_scale.exp() * scores
    return scores


@torch.inference_mode()
def score_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    images: List[Image.Image],
    all_texts: List[str],
    texts_per_item: int = 5,
    effective_model_type: str = "clip",
    text_max_len: int = 77,
    use_lowercase: bool = False,
) -> torch.Tensor:
    """
    Batch forward for N 个 (image, 5 texts)。
    images: 长度 N，第 i 个对应 all_texts[i*5 : (i+1)*5]
    Returns logits: (N, 5)，每行仅对应该图像的 5 个候选文本分数。
    """
    if not images or not all_texts:
        return torch.tensor([], device=device)
    if len(all_texts) != len(images) * texts_per_item:
        raise ValueError(
            f"all_texts length mismatch: got {len(all_texts)}, expected {len(images) * texts_per_item}"
        )

    all_texts = _maybe_lower_texts(all_texts, use_lowercase)
    core_model = _get_core_model(model)

    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16

    if effective_model_type in ("open_clip", "longclip", "pe"):
        image_tensors = torch.stack([processor["image_preprocess"](img) for img in images], dim=0).to(device)
        text_tokens = _to_device(processor["tokenizer"](all_texts), device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.encode_image(image_tensors)
            text_features = core_model.encode_text(text_tokens)
    else:
        image_inputs = processor(
            images=images,
            return_tensors="pt",
        )
        image_inputs = _to_device(image_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.get_image_features(**image_inputs)

        text_inputs = processor(
            text=all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_len,
        )
        text_inputs = _to_device(text_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            text_features = core_model.get_text_features(**text_inputs)

    image_features = image_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features.float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = core_model.logit_scale.exp() if hasattr(core_model, "logit_scale") else 1.0
    n = len(images)
    d = image_features.shape[-1]
    t = text_features.view(n, texts_per_item, d)            # (N, 5, D)
    i = image_features.unsqueeze(1)                         # (N, 1, D)
    # (N, 1, D) x (N, D, 5) -> (N, 1, 5) -> (N, 5)
    scores = torch.bmm(i, t.transpose(1, 2)).squeeze(1)
    return logit_scale * scores


def open_image_rgb(path: str) -> Image.Image:
    # Open via context manager to avoid lingering file handles.
    with Image.open(path) as img:
        return img.convert("RGB")


class ImageLevelDataset(Dataset):
    """Image-level dataset: each item carries one image and all its sets."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        try:
            image = open_image_rgb(row["image_path"])
        except Exception:
            image = None
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "sets": row["sets"],  # List[Tuple[tag, gt, negs]]
            "image": image,
        }


def build_image_rows(data_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pbar = tqdm(read_jsonl(data_path), desc="indexing")
    for idx, sample in enumerate(pbar, 1):
        if max_samples and idx > max_samples:
            break
        if "error" in sample and "sets" not in sample:
            continue
        try:
            sid, img_path, sets = extract_sets(sample)
        except Exception:
            continue
        if not os.path.exists(img_path):
            continue
        valid_sets: List[Tuple[str, str, List[str]]] = []
        for it in sets:
            try:
                tag, gt, negs = validate_set_item(it)
            except Exception:
                continue
            valid_sets.append((tag, gt, negs))
        if valid_sets:
            rows.append({
                "id": sid,
                "image_path": img_path,
                "sets": valid_sets,
            })
    return rows


def _replace_data_version(path: str, version: str) -> str:
    """
    Replace dataset version token in path, e.g. v2 -> v1/v3.
    """
    if version not in ("v1", "v2", "v3"):
        raise ValueError(f"Unsupported version: {version}")
    new_path, n = re.subn(r"v[123]", version, path, count=1)
    if n == 0:
        raise ValueError(
            f"Cannot infer data version from --data path: {path}. "
            "Please include one of v1/v2/v3 in the path."
        )
    return new_path


def _resolve_data_path(version: str, data_override: Optional[str]) -> str:
    """
    Resolve dataset path by version.
    Priority:
      1) --data override with v1/v2/v3 replacement
      2) built-in fixed paths
    """
    if version not in ("v1", "v2", "v3", "v2_gpt5", "v3_gpt5"):
        raise ValueError(f"Unsupported version: {version}")
    if data_override:
        return _replace_data_version(data_override, version)
    return DEFAULT_DATA_BY_VERSION[version]


def _version_results_dir(base_dir: str, version: str) -> str:
    base_name = os.path.basename(os.path.normpath(base_dir))
    if base_name == version:
        return base_dir
    return os.path.join(base_dir, version)


def _versioned_error_path(path: str, version: str) -> str:
    if not path:
        return path
    stem, ext = os.path.splitext(path)
    ext = ext or ".jsonl"
    return f"{stem}_{version}{ext}"


def collate_image_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Filter out bad image decodes while keeping output format stable.
    batch = [x for x in batch if x.get("image") is not None]
    if not batch:
        return {
            "ids": [],
            "image_paths": [],
            "sets": [],
            "images": [],
        }
    return {
        "ids": [x["id"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
        "sets": [x["sets"] for x in batch],
        "images": [x["image"] for x in batch],
    }


@torch.inference_mode()
def score_sets_for_image_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    images: List[Image.Image],
    sets_per_image: List[List[Tuple[str, str, List[str]]]],
    effective_model_type: str = "clip",
    text_max_len: int = 77,
    use_lowercase: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int, str, str, List[str]]]]:
    """
    Score all sets in one image batch.
    Returns:
      - scores: (M, 5) where M is total set count in this image batch
      - meta:   list of tuples (img_idx, tag, gt, negs) length M
    """
    if not images:
        return torch.empty(0, 5, device=device), []

    core_model = _get_core_model(model)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16

    # 1) Encode image features once per image.
    if effective_model_type in ("open_clip", "longclip", "pe"):
        image_tensors = torch.stack([processor["image_preprocess"](img) for img in images], dim=0).to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.encode_image(image_tensors)
    else:
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = _to_device(image_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            image_features = core_model.get_image_features(**image_inputs)
    image_features = image_features.float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 2) Flatten all sets (for all images in this batch) into text list.
    all_texts: List[str] = []
    set_meta: List[Tuple[int, str, str, List[str]]] = []
    owner_image_indices: List[int] = []
    for img_idx, sets in enumerate(sets_per_image):
        for tag, gt, negs in sets:
            candidates = _maybe_lower_texts([gt] + negs, use_lowercase)
            all_texts.extend(candidates)
            set_meta.append((img_idx, tag, gt, negs))
            owner_image_indices.append(img_idx)

    if not all_texts:
        return torch.empty(0, 5, device=device), []

    # 3) Encode all candidate texts once.
    if effective_model_type in ("open_clip", "longclip", "pe"):
        text_tokens = _to_device(processor["tokenizer"](all_texts), device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            text_features = core_model.encode_text(text_tokens)
    else:
        text_inputs = processor(
            text=all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_len,
        )
        text_inputs = _to_device(text_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            text_features = core_model.get_text_features(**text_inputs)
    text_features = text_features.float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4) For each set, score its own 5 candidates against its image.
    m = len(set_meta)
    d = image_features.shape[-1]
    t = text_features.view(m, 5, d)  # (M, 5, D)
    owners = torch.tensor(owner_image_indices, device=image_features.device, dtype=torch.long)
    i = image_features[owners].unsqueeze(1)  # (M, 1, D)
    scores = torch.bmm(i, t.transpose(1, 2)).squeeze(1)  # (M, 5)
    if hasattr(core_model, "logit_scale"):
        scores = core_model.logit_scale.exp() * scores
    return scores, set_meta


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate CLIP-style models on your 5-way sets (GT + 4 Neg) accuracy."
    )
    ap.add_argument(
        "--data",
        default="",
        help="Optional JSONL path override. If omitted, built-in v1/v2/v3 paths are used.",
    )
    ap.add_argument(
        "--data-version",
        default="v2",
        choices=["v1", "v2", "v3", "v2_gpt5", "v3_gpt5", "all"],
        help=(
            "Dataset version selector. "
            "'v1'/'v2'/'v3' are GPT-4 versions; 'v2_gpt5'/'v3_gpt5' are GPT-5 versions. "
            "If 'all', run v1/v2/v3 sequentially."
        ),
    )
    ap.add_argument("--model", required=True, help="Model repo/path (HF or custom, depending on --model-type).")
    ap.add_argument(
        "--processor",
        default=None,
        help="HF processor repo/path. Default: same as --model for HF backends. Ignored by open_clip/longclip/pe.",
    )
    ap.add_argument(
        "--model-type",
        default="auto",
        help="auto|clip|siglip|siglip2|metaclip|open_clip|longclip|pe",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Use cuda if available.")
    ap.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all visible CUDA devices via torch.nn.DataParallel.",
    )
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Number of (image, set) pairs per forward. Default 512 for speed; reduce if OOM.")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Number of DataLoader workers for image decode.")
    ap.add_argument("--batch-text", action="store_true",
                    help="(Optional) Process multiple sets per image in one forward by concatenating texts. "
                         "Ignored when using --batch-size > 1.")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit; otherwise eval first N samples.")
    ap.add_argument("--save-errors", default="", help="Optional output JSONL to save wrong predictions.")
    ap.add_argument("--topk", type=int, default=5, help="When saving errors, save topk ranked texts.")
    ap.add_argument(
        "--results-dir",
        default="/home/localadmin/bz/CLIP-R/eval/results/rclip",
        help="Directory to save txt results. Default: eval/results/rclip under this script.",
    )
    args = ap.parse_args()

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = args.results_dir or os.path.join(script_dir, "results", "rclip")
    os.makedirs(results_root, exist_ok=True)

    # Pre-check: skip if all requested versions already have results
    mt = "auto" if args.model_type is None else str(args.model_type).strip().lower()
    if mt in ("", "auto"):
        name_model_type = _infer_model_type(args.model)
    else:
        name_model_type = _infer_model_type(args.model_type)

    # NOTE: 'all' 目前只遍历 v1/v2/v3，gpt5 版本需单独跑
    versions = ["v1", "v2", "v3"] if args.data_version == "all" else [args.data_version]
    model_name_for_check = str(args.model).replace("/", "_").replace(":", "_")
    need_run = False
    for version in versions:
        data_path = _resolve_data_path(version, args.data or None)
        if not os.path.exists(data_path):
            continue
        results_dir = _version_results_dir(results_root, version)
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        txt_path = os.path.join(results_dir, f"rclip_results_{name_model_type}_{model_name_for_check}_{data_name}.txt")
        if not os.path.exists(txt_path):
            need_run = True
            break

    if not need_run:
        print(f"[SKIP] All requested versions already evaluated for model: {args.model}")
        return

    # load model/processor (only when at least one version needs running)
    loaded = load_model_and_processor(
        model_path=args.model,
        processor_path=args.processor,
        model_type=args.model_type,
        device=device,
    )
    model = loaded["model"]
    processor = loaded["processor"]
    model_type = loaded["model_type"]
    effective_model_type = loaded["effective_model_type"]
    text_max_len = loaded["text_max_len"]
    use_lowercase = loaded["use_lowercase"]

    print(f"Model type: {loaded['model_type']} (effective: {effective_model_type})")
    if use_lowercase:
        print("Text lowercase: enabled (siglip2)")

    # optional multi-GPU (DataParallel) when using CUDA
    if device.type == "cuda" and args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)
    model.eval()

    for version in versions:
        data_path = _resolve_data_path(version, args.data or None)
        if not os.path.exists(data_path):
            print(f"[WARN] Skip {version}: data file not found: {data_path}")
            continue
        results_dir = _version_results_dir(results_root, version)
        os.makedirs(results_dir, exist_ok=True)
        model_name = str(args.model).replace("/", "_").replace(":", "_")
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        txt_path = os.path.join(results_dir, f"rclip_results_{model_type}_{model_name}_{data_name}.txt")
        if os.path.exists(txt_path):
            print(f"[SKIP] {version}: results already exist: {txt_path}")
            continue
        print(f"\n===== Running dataset {version}: {data_path} =====")

        # build image-level rows and dataloader
        image_rows = build_image_rows(data_path, max_samples=args.max_samples)
        dataset = ImageLevelDataset(image_rows)
        dataloader = DataLoader(
            dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            num_workers=max(0, args.num_workers),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_image_batch,
        )

        # stats
        total_sets = 0
        correct_sets = 0
        per_tag_total: Dict[str, int] = {}
        per_tag_correct: Dict[str, int] = {}
        error_rows: List[Dict[str, Any]] = []
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"batches-{version}")
        for batch in pbar:
            if not batch["images"]:
                continue
            logits, meta = score_sets_for_image_batch(
                model,
                processor,
                device,
                batch["images"],
                batch["sets"],
                effective_model_type=effective_model_type,
                text_max_len=text_max_len,
                use_lowercase=use_lowercase,
            )
            if logits.numel() == 0:
                continue
            logits = logits.float().cpu()
            for i, (img_idx, tag, gt, negs) in enumerate(meta):
                sid = batch["ids"][img_idx]
                img_path = batch["image_paths"][img_idx]
                scores = logits[i]
                pred = int(torch.argmax(scores).item())
                is_correct = pred == 0
                total_sets += 1
                correct_sets += int(is_correct)
                per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                per_tag_correct[tag] = per_tag_correct.get(tag, 0) + int(is_correct)
                if (not is_correct) and args.save_errors:
                    texts = [gt] + negs
                    topk = min(args.topk, 5)
                    order = torch.argsort(scores, descending=True)[:topk].tolist()
                    error_rows.append({
                        "id": sid,
                        "image_path": img_path,
                        "tag": tag,
                        "gt": gt,
                        "neg": negs,
                        "pred_index": pred,
                        "pred_text": texts[pred],
                        "scores": [float(scores[j]) for j in range(5)],
                        "topk": [{"idx": j, "text": texts[j], "score": float(scores[j])} for j in order],
                    })
            acc = (correct_sets / total_sets) if total_sets else 0.0
            pbar.set_postfix({"set_acc": f"{acc:.4f}", "sets": total_sets})

        # print report
        overall_acc = (correct_sets / total_sets) if total_sets else 0.0
        print("\n=== Results ===")
        print(f"Version: {version}")
        print(f"Total sets: {total_sets}")
        print(f"Correct sets: {correct_sets}")
        print(f"Set-level Acc (GT ranked #1 among 5): {overall_acc:.6f}")

        print("\n--- Per-tag Acc ---")
        for tag in sorted(per_tag_total.keys()):
            t = per_tag_total[tag]
            c = per_tag_correct.get(tag, 0)
            print(f"{tag:>10s}: {c}/{t} = {c/t:.6f}")

        # Save txt report (default behavior)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== Results ===\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Model Type: {model_type} (effective: {effective_model_type})\n")
            f.write(f"Processor: {args.processor or args.model}\n")
            f.write(f"Data: {data_path}\n")
            f.write(f"Dataset Version: {version}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Num Workers: {args.num_workers}\n")
            f.write(f"Max Samples: {args.max_samples}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total sets: {total_sets}\n")
            f.write(f"Correct sets: {correct_sets}\n")
            f.write(f"Set-level Acc (GT ranked #1 among 5): {overall_acc:.6f}\n")
            f.write("\n--- Per-tag Acc ---\n")
            for tag in sorted(per_tag_total.keys()):
                t = per_tag_total[tag]
                c = per_tag_correct.get(tag, 0)
                f.write(f"{tag:>10s}: {c}/{t} = {c/t:.6f}\n")
        print(f"\nSaved txt results to: {txt_path}")

        # save errors
        if args.save_errors:
            versioned_errors = _versioned_error_path(args.save_errors, version)
            write_jsonl(versioned_errors, error_rows)
            print(f"\nSaved wrong cases to: {versioned_errors}  (rows={len(error_rows)})")


if __name__ == "__main__":
    main()
