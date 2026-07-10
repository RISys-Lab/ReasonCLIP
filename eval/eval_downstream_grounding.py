#!/usr/bin/env python3
"""Zero-shot candidate-box grounding on REFER RefCOCO/RefCOCO+."""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import (  # noqa: E402
    load_model_bundle,
    processor_geometry_summary,
    safe_model_name,
    write_json,
)


@dataclass
class CandidateGroundingMetrics:
    benchmark: str
    dataset: str
    split: str
    split_by: str
    model_name: str
    model_id: str
    processor_id: str
    tokenizer_id: str
    family: str
    refs: int
    expressions: int
    images: int
    missing_images: int
    acc_ann: float
    acc_iou_0_5: float
    mean_iou: float
    mean_candidates: float
    text_template: str
    processor_geometry: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RefCOCO candidate-box grounding")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["refcoco", "refcoco+"], required=True)
    parser.add_argument("--split", choices=["val", "test", "testA", "testB"], default="testA")
    parser.add_argument("--split-by", default="unc")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-refs", type=int, default=None)
    parser.add_argument("--hf-fallback", action="store_true")
    parser.add_argument("--cache-dir", default="/home/localadmin/.cache/huggingface/datasets")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0, help="Accepted for runner compatibility; unused")
    parser.add_argument("--seed", type=int, default=7, help="Accepted for runner compatibility")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results" / "downstream")
    parser.add_argument("--text-template", default="{}", help="Format string applied to each referring expression")
    return parser.parse_args()


def load_refs(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def keep_split(ref_split: str, requested: str) -> bool:
    return str(ref_split).lower() == requested.lower()


def xywh_to_xyxy(box) -> np.ndarray:
    x, y, w, h = [float(value) for value in box]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    left_top = np.maximum(a[:2], b[:2])
    right_bottom = np.minimum(a[2:], b[2:])
    wh = np.maximum(right_bottom - left_top, 0.0)
    inter = float(wh[0] * wh[1])
    area_a = float(max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0))
    area_b = float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))
    return inter / max(area_a + area_b - inter, 1e-6)


def crop_box(image: Image.Image, box_xywh, pad: float) -> Image.Image:
    width, height = image.size
    x, y, w, h = [float(value) for value in box_xywh]
    px = w * pad
    py = h * pad
    x1 = max(0.0, x - px)
    y1 = max(0.0, y - py)
    x2 = min(float(width), x + w + px)
    y2 = min(float(height), y + h + py)
    if x2 <= x1 or y2 <= y1:
        return image.convert("RGB")
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def load_hf_image_cache(dataset: str, split: str, cache_dir: str) -> dict[int, Image.Image]:
    hf_name = "lmms-lab/RefCOCOplus" if dataset == "refcoco+" else "lmms-lab/RefCOCO"
    ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)
    images: dict[int, Image.Image] = {}
    for item in ds:
        stem = Path(item["file_name"]).stem
        parts = stem.split("_")
        if len(parts) >= 3:
            image_id = int(parts[2])
            images.setdefault(image_id, item["image"].convert("RGB"))
    return images


def load_hf_arrow_image_cache(dataset: str, split: str, cache_dir: str) -> dict[int, Image.Image]:
    """Read cached HF arrow files directly when dataset metadata is version-incompatible."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError:
        return {}
    cache_name = "lmms-lab___ref_coc_oplus" if dataset == "refcoco+" else "lmms-lab___ref_coco"
    paths = sorted(Path(cache_dir).glob(f"{cache_name}/**/*-{split}*.arrow"))
    if not paths:
        return {}
    images: dict[int, Image.Image] = {}
    for arrow_path in paths:
        with pa.memory_map(str(arrow_path), "r") as source:
            table = ipc.RecordBatchStreamReader(source).read_all()
        for row in table.to_pylist():
            file_name = row.get("file_name") or (row.get("image") or {}).get("path")
            if not file_name:
                continue
            parts = Path(file_name).stem.split("_")
            if len(parts) < 3:
                continue
            image_id = int(parts[2])
            if image_id in images:
                continue
            image_bytes = (row.get("image") or {}).get("bytes")
            if image_bytes:
                images[image_id] = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return images


def open_image_zip(path: Path):
    if not path.exists():
        return None
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        print(f"Local image zip is unreadable and will be skipped: {path}: {exc}", file=sys.stderr)
        return None


def tokenizer_max_len(tokenizer) -> int:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len > 100000:
        return 64 if "siglip" in tokenizer.__class__.__name__.lower() else 77
    return int(max_len)


def encode_images(model, processor, images: list[Image.Image], device: str, batch_size: int) -> torch.Tensor:
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        features.append(F.normalize(out.float(), dim=-1).detach().cpu())
    return torch.cat(features, dim=0)


def encode_texts(model, tokenizer, texts: list[str], device: str, batch_size: int) -> torch.Tensor:
    features = []
    max_len = tokenizer_max_len(tokenizer)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        features.append(F.normalize(out.float(), dim=-1).detach().cpu())
    return torch.cat(features, dim=0)


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(
        args.model_id,
        processor_id=args.processor_id,
        tokenizer_id=args.tokenizer_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    model = bundle["model"]
    processor = bundle["processor"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]

    dataset_dir = args.data_root / args.dataset
    image_dir = args.data_root / "images" / "mscoco" / "images" / "train2014"
    image_zip_path = image_dir.with_suffix(".zip")
    image_zip = open_image_zip(image_zip_path)
    refs_path = dataset_dir / f"refs({args.split_by}).p"
    instances_path = dataset_dir / "instances.json"
    if not refs_path.exists() or not instances_path.exists():
        raise FileNotFoundError(f"Missing REFER files under {dataset_dir}")
    refs = [ref for ref in load_refs(refs_path) if keep_split(ref["split"], args.split)]
    if args.max_refs is not None:
        refs = refs[: args.max_refs]
    with instances_path.open("r", encoding="utf-8") as handle:
        instances = json.load(handle)
    anns = {int(ann["id"]): ann for ann in instances["annotations"]}
    imgs = {int(img["id"]): img for img in instances["images"]}
    image_to_anns: dict[int, list[dict]] = {}
    for ann in instances["annotations"]:
        image_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

    hf_images = {}
    if args.hf_fallback:
        try:
            hf_images = load_hf_image_cache(args.dataset, args.split, args.cache_dir)
        except Exception as exc:
            print(
                f"HF image fallback unavailable for {args.dataset}/{args.split}: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            hf_images = load_hf_arrow_image_cache(args.dataset, args.split, args.cache_dir)
            if hf_images:
                print(f"Loaded {len(hf_images)} images from cached HF arrow files", file=sys.stderr)
    image_feature_cache: dict[int, tuple[list[int], np.ndarray, torch.Tensor]] = {}
    correct_ann = 0
    correct_iou = 0
    iou_sum = 0.0
    expr_count = 0
    candidate_counts = []
    missing = 0

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "processor_id": bundle["processor_id"],
                "tokenizer_id": bundle["tokenizer_id"],
                "family": bundle["family"],
                "dataset": args.dataset,
                "split": args.split,
                "refs": len(refs),
                "device": device,
                "tokenizer_max_length": tokenizer_max_len(tokenizer),
                "text_template": args.text_template,
                "processor_geometry": processor_geometry_summary(processor),
            },
            sort_keys=True,
        )
    )

    for ref in tqdm(refs, desc=f"{args.dataset} {args.split}"):
        image_id = int(ref["image_id"])
        ann_id = int(ref["ann_id"])
        if image_id not in image_feature_cache:
            image_path = image_dir / imgs[image_id]["file_name"]
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
            elif image_id in hf_images:
                image = hf_images[image_id]
            elif image_zip is not None:
                image = None
                for member in (imgs[image_id]["file_name"], f"train2014/{imgs[image_id]['file_name']}"):
                    try:
                        with image_zip.open(member) as handle:
                            image = Image.open(handle).convert("RGB")
                        break
                    except KeyError:
                        continue
                if image is None:
                    missing += 1
                    continue
            else:
                missing += 1
                continue
            cand_anns = image_to_anns.get(image_id, [])
            if not cand_anns:
                continue
            cand_ids = [int(ann["id"]) for ann in cand_anns]
            cand_boxes = np.stack([xywh_to_xyxy(ann["bbox"]) for ann in cand_anns], axis=0)
            crops = [crop_box(image, ann["bbox"], pad=0.08) for ann in cand_anns]
            crop_feats = encode_images(model, processor, crops, device, args.batch_size)
            image_feature_cache[image_id] = (cand_ids, cand_boxes, crop_feats)

        cand_ids, cand_boxes, crop_feats = image_feature_cache[image_id]
        if ann_id not in cand_ids:
            continue
        target_box = xywh_to_xyxy(anns[ann_id]["bbox"])
        candidate_counts.append(len(cand_ids))
        texts = [args.text_template.format(sentence["sent"]) for sentence in ref["sentences"]]
        text_feats = encode_texts(model, tokenizer, texts, device, args.batch_size)
        scores = text_feats @ crop_feats.T
        best = scores.argmax(dim=1).tolist()
        for best_idx in best:
            pred_ann_id = cand_ids[best_idx]
            pred_box = cand_boxes[best_idx]
            score_iou = iou_xyxy(pred_box, target_box)
            correct_ann += int(pred_ann_id == ann_id)
            correct_iou += int(score_iou >= 0.5)
            iou_sum += score_iou
            expr_count += 1

    if expr_count == 0:
        raise RuntimeError("No RefCOCO expressions were evaluated")
    model_name = args.model_name or safe_model_name(args.model_id)
    metrics = CandidateGroundingMetrics(
        benchmark="grounding",
        dataset=args.dataset,
        split=args.split,
        split_by=args.split_by,
        model_name=model_name,
        model_id=args.model_id,
        processor_id=bundle["processor_id"],
        tokenizer_id=bundle["tokenizer_id"],
        family=bundle["family"],
        refs=len(refs) - missing,
        expressions=expr_count,
        images=len(image_feature_cache),
        missing_images=missing,
        acc_ann=correct_ann / expr_count,
        acc_iou_0_5=correct_iou / expr_count,
        mean_iou=iou_sum / expr_count,
        mean_candidates=float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        text_template=args.text_template,
        processor_geometry=processor_geometry_summary(processor),
    )
    safe_dataset = args.dataset.replace("+", "plus")
    out = args.out_dir / "grounding" / f"{safe_dataset}_{args.split}_{safe_model_name(model_name)}.json"
    write_json(out, metrics)
    print(json.dumps(asdict(metrics), sort_keys=True))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
