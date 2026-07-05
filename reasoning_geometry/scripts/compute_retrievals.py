#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from common import DEFAULT_PROMPTS, IMAGE_MODELS, l2_normalize, parse_model_keys, read_json, read_jsonl, write_json


def embed_prompts(model_key: str, prompts: List[str], device: str, local_files_only: bool, no_fp16: bool) -> np.ndarray:
    cfg = IMAGE_MODELS[model_key]
    print(f"loading text encoder for {model_key}: {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(cfg["processor_id"], local_files_only=local_files_only)
    kwargs = {"local_files_only": local_files_only}
    if device.startswith("cuda") and not no_fp16:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModel.from_pretrained(cfg["model_id"], **kwargs).to(device)
    model.eval()
    inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        feats = model.get_text_features(**inputs)
    arr = l2_normalize(feats.float().cpu().numpy())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return arr


def load_prompts(path: str | None):
    if not path:
        return DEFAULT_PROMPTS
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    return obj["prompts"]


def main():
    parser = argparse.ArgumentParser(description="Precompute text prompt to image retrieval for the explorer.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--prompts", default=None, help="Optional JSON prompt list")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.annotations)
    ids = [str(r["image_id"]) for r in rows]
    emb_dir = Path(args.image_embedding_dir)
    saved_ids = read_json(emb_dir / "ids.json")
    if list(saved_ids) != ids:
        raise ValueError("annotation ids do not match image embedding ids")
    prompts = load_prompts(args.prompts)
    prompt_texts = [p["text"] for p in prompts]
    output: Dict[str, object] = {"prompts": prompts, "models": {}}
    for model_key in parse_model_keys(args.models):
        img = np.load(emb_dir / f"{model_key}.npy").astype(np.float32)
        txt = embed_prompts(model_key, prompt_texts, args.device, args.local_files_only, args.no_fp16)
        if img.shape[1] != txt.shape[1]:
            raise ValueError(f"dimension mismatch for {model_key}: image {img.shape}, text {txt.shape}")
        sims = txt @ img.T
        model_out = {}
        for p_idx, prompt in enumerate(prompts):
            scores = sims[p_idx]
            k = min(args.top_k, len(scores))
            top = np.argpartition(-scores, kth=k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            model_out[prompt["id"]] = [{"index": int(i), "image_id": ids[int(i)], "score": float(scores[int(i)])} for i in top]
        output["models"][model_key] = model_out
        print(f"computed retrievals for {model_key}")
    write_json(args.output, output)
    print(f"wrote retrievals to {args.output}")


if __name__ == "__main__":
    main()
