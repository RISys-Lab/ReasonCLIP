#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from common import IMAGE_MODELS, l2_normalize, parse_model_keys, read_jsonl, save_numpy, write_json


def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def same_ids(path: Path, ids: List[str]) -> bool:
    if not path.exists():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return list(old) == list(ids)


def model_dtype(device: str, no_fp16: bool):
    if device.startswith("cuda") and not no_fp16:
        return torch.float16
    return None


def embed_one_model(model_key: str, rows, ids: List[str], out_dir: Path, batch_size: int, device: str, local_files_only: bool, no_fp16: bool, resume: bool):
    cfg = IMAGE_MODELS[model_key]
    out_path = out_dir / f"{model_key}.npy"
    meta_path = out_dir / f"{model_key}.meta.json"
    ids_path = out_dir / "ids.json"
    if resume and out_path.exists() and same_ids(ids_path, ids):
        print(f"skip existing {model_key}: {out_path}")
        return
    print(f"loading {model_key}: {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(cfg["processor_id"], local_files_only=local_files_only)
    kwargs = {"local_files_only": local_files_only}
    dtype = model_dtype(device, no_fp16)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(cfg["model_id"], **kwargs).to(device)
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        images = [load_image(str(r["image_path"])) for r in batch_rows]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
        chunks.append(feats.float().cpu().numpy())
        print(f"{model_key}: {min(start + batch_size, len(rows))}/{len(rows)}")
    embeddings = l2_normalize(np.concatenate(chunks, axis=0))
    save_numpy(out_path, embeddings)
    write_json(ids_path, ids)
    write_json(meta_path, {**cfg, "model_key": model_key, "num_rows": len(rows), "dim": int(embeddings.shape[1]), "normalized": True, "local_files_only": local_files_only})
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"wrote {model_key} embeddings {embeddings.shape} to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed images with baseline/S1/S2 CLIP and SigLIP variants.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="all", help="Comma-separated model keys or all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    ids = [str(r["image_id"]) for r in rows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in parse_model_keys(args.models):
        embed_one_model(key, rows, ids, out_dir, args.batch_size, args.device, args.local_files_only, args.no_fp16, args.resume)


if __name__ == "__main__":
    main()
