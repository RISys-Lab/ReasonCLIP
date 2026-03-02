#!/usr/bin/env python3
#
# API Key: export GEMINI_API_KEY="*"
#
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Tuple, Optional

import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from torch.utils.data import Dataset, DataLoader


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
        return {
            "id": row["id"],
            "image_path": row["image_path"],
            "sets": row["sets"],  # List[Tuple[tag, gt, negs]]
            "image": open_image_rgb(row["image_path"]),
        }


def build_image_rows(data_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pbar = tqdm(read_jsonl(data_path), desc="indexing")
    for idx, sample in enumerate(pbar, 1):
        if max_samples and idx > max_samples:
            break
        sid, img_path, sets = extract_sets(sample)
        valid_sets: List[Tuple[str, str, List[str]]] = []
        for it in sets:
            tag, gt, negs = validate_set_item(it)
            valid_sets.append((tag, gt, negs))
        rows.append({
            "id": sid,
            "image_path": img_path,
            "sets": valid_sets,
        })
    return rows


def collate_image_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [x["id"] for x in batch],
        "image_paths": [x["image_path"] for x in batch],
        "sets": [x["sets"] for x in batch],
        "images": [x["image"] for x in batch],
    }


@torch.inference_mode()
def encode_image_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    images: List[Image.Image],
    effective_model_type: str,
) -> torch.Tensor:
    core_model = _get_core_model(model)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16
    if effective_model_type in ("open_clip", "longclip", "pe"):
        image_tensors = torch.stack([processor["image_preprocess"](img) for img in images], dim=0).to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            feats = core_model.encode_image(image_tensors)
    else:
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = _to_device(image_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            feats = core_model.get_image_features(**image_inputs)
    feats = feats.float()
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.inference_mode()
def encode_text_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    texts: List[str],
    effective_model_type: str,
    text_max_len: int,
    use_lowercase: bool,
) -> torch.Tensor:
    core_model = _get_core_model(model)
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.float16
    texts = _maybe_lower_texts(texts, use_lowercase)
    if effective_model_type in ("open_clip", "longclip", "pe"):
        text_tokens = _to_device(processor["tokenizer"](texts), device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            feats = core_model.encode_text(text_tokens)
    else:
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_len,
        )
        text_inputs = _to_device(text_inputs, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            feats = core_model.get_text_features(**text_inputs)
    feats = feats.float()
    return feats / feats.norm(dim=-1, keepdim=True)


def build_retrieval_lists(
    image_rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[List[int]], List[str], List[int]]:
    # I2T gallery: GT-only text pool, 5 GT per image.
    i2t_texts: List[str] = []
    image_pos_text_ids: List[List[int]] = []
    # T2I queries: all GT texts, each maps to its owner image index.
    t2i_query_texts: List[str] = []
    t2i_query_gt_img_idx: List[int] = []

    for img_idx, row in enumerate(image_rows):
        pos_ids: List[int] = []
        for _tag, gt, negs in row["sets"]:
            base = len(i2t_texts)
            i2t_texts.append(gt)
            pos_ids.append(base)
            t2i_query_texts.append(gt)     # only GT texts for T2I queries
            t2i_query_gt_img_idx.append(img_idx)
        image_pos_text_ids.append(pos_ids)
    return i2t_texts, image_pos_text_ids, t2i_query_texts, t2i_query_gt_img_idx


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Retrieval eval on RCLIP: GT-only text pool (5000x5) for both I2T and T2I."
    )
    ap.add_argument("--data", required=True, help="JSONL dataset path (each line has image_path and sets).")
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
                    help="Image batch size for image encoding.")
    ap.add_argument("--text-batch-size", type=int, default=2048,
                    help="Text batch size for text encoding.")
    ap.add_argument("--sim-chunk-size", type=int, default=512,
                    help="Query chunk size when computing similarity matrix.")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Number of DataLoader workers for image decode.")
    ap.add_argument("--k-values", type=str, default="1,5,10",
                    help="Comma-separated retrieval k values, e.g. 1,5,10.")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit; otherwise eval first N samples.")
    ap.add_argument(
        "--results-dir",
        default="/home/localadmin/bz/CLIP-R/eval/results/rclip/v3_retrieval",
        help="Directory to save txt results. Default: eval/results/rclip under this script.",
    )
    args = ap.parse_args()

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(script_dir, "results", "rclip")
    os.makedirs(results_dir, exist_ok=True)

    # Pre-check: skip before loading model
    mt = "auto" if args.model_type is None else str(args.model_type).strip().lower()
    if mt in ("", "auto"):
        name_model_type = _infer_model_type(args.model)
    else:
        name_model_type = _infer_model_type(args.model_type)

    model_name = str(args.model).replace("/", "_").replace(":", "_")
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    txt_path = os.path.join(results_dir, f"rclip_results_{name_model_type}_{model_name}_{data_name}.txt")
    if os.path.exists(txt_path):
        print(f"[SKIP] Results already exist: {txt_path}")
        return

    # load model/processor (only when needed)
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

    # build image-level rows and dataloader
    image_rows = build_image_rows(args.data, max_samples=args.max_samples)
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

    # 1) Encode all images
    image_feats_parts: List[torch.Tensor] = []
    pbar = tqdm(dataloader, total=len(dataloader), desc="encode_images")
    for batch in pbar:
        feats = encode_image_batch(
            model,
            processor,
            device,
            batch["images"],
            effective_model_type=effective_model_type,
        )
        image_feats_parts.append(feats.cpu())
    image_feats = torch.cat(image_feats_parts, dim=0)  # (N_img, D)

    # 2) Build retrieval lists
    i2t_texts, image_pos_text_ids, t2i_query_texts, t2i_query_gt_img_idx = build_retrieval_lists(image_rows)

    # 3) Encode I2T text gallery (GT-only, 5000*5)
    i2t_text_feats_parts: List[torch.Tensor] = []
    for s in tqdm(range(0, len(i2t_texts), args.text_batch_size), desc="encode_i2t_texts"):
        e = s + args.text_batch_size
        feats = encode_text_batch(
            model=model,
            processor=processor,
            device=device,
            texts=i2t_texts[s:e],
            effective_model_type=effective_model_type,
            text_max_len=text_max_len,
            use_lowercase=use_lowercase,
        )
        i2t_text_feats_parts.append(feats.cpu())
    i2t_text_feats = torch.cat(i2t_text_feats_parts, dim=0)  # (N_img*25, D)

    # 4) Encode T2I text queries (5000*5 GT)
    t2i_text_feats_parts: List[torch.Tensor] = []
    for s in tqdm(range(0, len(t2i_query_texts), args.text_batch_size), desc="encode_t2i_queries"):
        e = s + args.text_batch_size
        feats = encode_text_batch(
            model=model,
            processor=processor,
            device=device,
            texts=t2i_query_texts[s:e],
            effective_model_type=effective_model_type,
            text_max_len=text_max_len,
            use_lowercase=use_lowercase,
        )
        t2i_text_feats_parts.append(feats.cpu())
    t2i_text_feats = torch.cat(t2i_text_feats_parts, dim=0)  # (N_img*5, D)

    # 5) Retrieval metrics
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    max_k = max(k_values)
    if device.type == "cuda":
        image_feats_dev = image_feats.to(device)
        i2t_text_feats_dev = i2t_text_feats.to(device)
        t2i_text_feats_dev = t2i_text_feats.to(device)
    else:
        image_feats_dev = image_feats
        i2t_text_feats_dev = i2t_text_feats
        t2i_text_feats_dev = t2i_text_feats

    i2t_hits = {k: 0 for k in k_values}
    i2t_ranks: List[int] = []
    for s in tqdm(range(0, image_feats_dev.shape[0], args.sim_chunk_size), desc="i2t_retrieval"):
        e = min(s + args.sim_chunk_size, image_feats_dev.shape[0])
        sim = image_feats_dev[s:e] @ i2t_text_feats_dev.t()  # (b, N_text)
        topk = torch.topk(sim, k=max_k, dim=1).indices.cpu()
        sim_cpu = sim.float().cpu()
        for i in range(topk.shape[0]):
            pos_ids = image_pos_text_ids[s + i]
            pos = set(pos_ids)
            row = topk[i].tolist()
            for k in k_values:
                if any(idx in pos for idx in row[:k]):
                    i2t_hits[k] += 1
            # rank of first relevant text among all texts
            pos_scores = sim_cpu[i, pos_ids]
            best_pos_score = torch.max(pos_scores).item()
            rank = int((sim_cpu[i] > best_pos_score).sum().item() + 1)
            i2t_ranks.append(rank)

    t2i_hits = {k: 0 for k in k_values}
    t2i_ranks: List[int] = []
    gt_img = torch.tensor(t2i_query_gt_img_idx, dtype=torch.long)
    for s in tqdm(range(0, t2i_text_feats_dev.shape[0], args.sim_chunk_size), desc="t2i_retrieval"):
        e = min(s + args.sim_chunk_size, t2i_text_feats_dev.shape[0])
        sim = t2i_text_feats_dev[s:e] @ image_feats_dev.t()  # (b, N_img)
        topk = torch.topk(sim, k=max_k, dim=1).indices.cpu()
        sim_cpu = sim.float().cpu()
        gt = gt_img[s:e]
        for k in k_values:
            hit = (topk[:, :k] == gt.unsqueeze(1)).any(dim=1).sum().item()
            t2i_hits[k] += int(hit)
        for i in range(sim_cpu.shape[0]):
            gt_idx = int(gt[i].item())
            gt_score = sim_cpu[i, gt_idx].item()
            rank = int((sim_cpu[i] > gt_score).sum().item() + 1)
            t2i_ranks.append(rank)

    n_img = image_feats.shape[0]
    n_tq = t2i_text_feats.shape[0]
    i2t_r = {k: i2t_hits[k] / n_img for k in k_values}
    t2i_r = {k: t2i_hits[k] / n_tq for k in k_values}
    i2t_ranks_sorted = sorted(i2t_ranks)
    t2i_ranks_sorted = sorted(t2i_ranks)
    i2t_mean_rank = sum(i2t_ranks) / len(i2t_ranks)
    t2i_mean_rank = sum(t2i_ranks) / len(t2i_ranks)
    i2t_median_rank = i2t_ranks_sorted[len(i2t_ranks_sorted) // 2]
    t2i_median_rank = t2i_ranks_sorted[len(t2i_ranks_sorted) // 2]

    print("\n=== Retrieval Results ===")
    print(f"Images: {n_img}")
    print(f"I2T text gallery size: {len(i2t_texts)}")
    print(f"T2I query text size: {n_tq}")
    for k in k_values:
        print(f"I2T R@{k}: {i2t_r[k]:.6f}")
    print(f"I2T Mean Rank: {i2t_mean_rank:.4f}")
    print(f"I2T Median Rank: {i2t_median_rank}")
    for k in k_values:
        print(f"T2I R@{k}: {t2i_r[k]:.6f}")
    print(f"T2I Mean Rank: {t2i_mean_rank:.4f}")
    print(f"T2I Median Rank: {t2i_median_rank}")

    # Save txt report (default behavior)
    model_name = str(args.model).replace("/", "_").replace(":", "_")
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    txt_path = os.path.join(results_dir, f"rclip_results_{model_type}_{model_name}_{data_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Retrieval Results ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Model Type: {model_type} (effective: {effective_model_type})\n")
        f.write(f"Processor: {args.processor or args.model}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Image Batch Size: {args.batch_size}\n")
        f.write(f"Text Batch Size: {args.text_batch_size}\n")
        f.write(f"Sim Chunk Size: {args.sim_chunk_size}\n")
        f.write(f"K Values: {args.k_values}\n")
        f.write(f"Num Workers: {args.num_workers}\n")
        f.write(f"Max Samples: {args.max_samples}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Images: {n_img}\n")
        f.write(f"I2T text gallery size: {len(i2t_texts)}\n")
        f.write(f"T2I query text size: {n_tq}\n")
        for k in k_values:
            f.write(f"I2T R@{k}: {i2t_r[k]:.6f}\n")
        f.write(f"I2T Mean Rank: {i2t_mean_rank:.4f}\n")
        f.write(f"I2T Median Rank: {i2t_median_rank}\n")
        for k in k_values:
            f.write(f"T2I R@{k}: {t2i_r[k]:.6f}\n")
        f.write(f"T2I Mean Rank: {t2i_mean_rank:.4f}\n")
        f.write(f"T2I Median Rank: {t2i_median_rank}\n")
    print(f"\nSaved txt results to: {txt_path}")


if __name__ == "__main__":
    main()
