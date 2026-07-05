#!/usr/bin/env python3
import argparse
import os
import hashlib
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from common import l2_normalize, read_jsonl, save_numpy, simple_tokens, write_json


def resolve_model_path(model_id: str, local_files_only: bool) -> str:
    if not local_files_only:
        return model_id
    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:
        return model_id


def reasoning_text(row: Dict[str, object]) -> str:
    caps = row.get("reasoning_captions") or []
    texts: List[str] = []
    for cap in caps:
        if isinstance(cap, dict):
            texts.append(str(cap.get("caption") or ""))
        else:
            texts.append(str(cap))
    return " ".join(t for t in texts if t).strip() or str(row.get("descriptive_caption") or row.get("source_caption") or "")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def pool_output(output, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return output.last_hidden_state[:, 0]
    if pooling == "pooler" and getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    return mean_pool(output.last_hidden_state, attention_mask)


def embed_hf(texts: List[str], model_id: str, batch_size: int, device: str, local_files_only: bool, pooling: str) -> np.ndarray:
    resolved_model = resolve_model_path(model_id, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(resolved_model, local_files_only=local_files_only).to(device)
    model.eval()
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs)
            pooled = pool_output(output, inputs["attention_mask"], pooling)
        outs.append(pooled.float().cpu().numpy())
        print(f"caption hf embeddings: {min(start + batch_size, len(texts))}/{len(texts)}")
    return l2_normalize(np.concatenate(outs, axis=0))


def stable_hash(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return idx, sign


def embed_hashing(texts: List[str], dim: int) -> np.ndarray:
    arr = np.zeros((len(texts), dim), dtype=np.float32)
    dfs: Dict[str, int] = {}
    doc_tokens: List[List[str]] = []
    for text in texts:
        toks = simple_tokens(text)
        doc_tokens.append(toks)
        for tok in set(toks):
            dfs[tok] = dfs.get(tok, 0) + 1
    n = max(1, len(texts))
    for row_idx, toks in enumerate(doc_tokens):
        counts: Dict[str, int] = {}
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, count in counts.items():
            idx, sign = stable_hash(tok, dim)
            idf = np.log((1 + n) / (1 + dfs.get(tok, 0))) + 1.0
            arr[row_idx, idx] += sign * count * idf
    return l2_normalize(arr)


def main():
    parser = argparse.ArgumentParser(description="Embed reasoning captions with an independent language-only model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=["hf", "hashing"], default="hf")
    parser.add_argument("--model", default="bert-base-uncased", help="HF language model for --method hf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pooling", choices=["mean", "cls", "pooler"], default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hash-dim", type=int, default=2048)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    ids = [str(r["image_id"]) for r in rows]
    texts = [reasoning_text(r) for r in rows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "hf":
        embeddings = embed_hf(texts, args.model, args.batch_size, args.device, args.local_files_only, args.pooling)
    else:
        embeddings = embed_hashing(texts, args.hash_dim)
    save_numpy(out_dir / "embeddings.npy", embeddings)
    write_json(out_dir / "ids.json", ids)
    write_json(out_dir / "captions.json", texts)
    write_json(out_dir / "meta.json", {"method": args.method, "model": args.model if args.method == "hf" else f"hashing-{args.hash_dim}", "pooling": args.pooling if args.method == "hf" else None, "num_rows": len(rows), "dim": int(embeddings.shape[1]), "normalized": True})
    print(f"wrote caption embeddings {embeddings.shape} to {out_dir}")


if __name__ == "__main__":
    main()
