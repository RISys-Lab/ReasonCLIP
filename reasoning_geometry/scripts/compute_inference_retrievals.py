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

from common import IMAGE_MODELS, l2_normalize, parse_model_keys, read_json, read_jsonl, write_json


def load_concepts(path: str | None) -> List[Dict[str, str]]:
    if not path:
        raise ValueError("--concepts is required")
    obj = read_json(path)
    concepts = obj["prompts"] if isinstance(obj, dict) and "prompts" in obj else obj
    out = []
    for item in concepts:
        prompt = item.get("retrieval_prompt") or item.get("text")
        label = item.get("display_label") or item.get("label") or prompt
        cid = item.get("id")
        if not cid or not prompt:
            raise ValueError(f"Invalid concept entry: {item}")
        concept = {"id": str(cid), "display_label": str(label), "retrieval_prompt": str(prompt)}
        for key in ["category", "category_label"]:
            if item.get(key):
                concept[key] = str(item[key])
        out.append(concept)
    return out


def family_stage_keys(model_keys: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for key in model_keys:
        cfg = IMAGE_MODELS[key]
        out.setdefault(cfg["family"], {})[cfg["stage"]] = key
    return out


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


def ranked_hits(scores: np.ndarray, ids: List[str], top_k: int) -> tuple[List[int], np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    top = order[:top_k].astype(np.int32)
    return top.tolist(), ranks, order


def round_float(x: float) -> float:
    return round(float(x), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute inference-only text-to-image retrievals for explorer v3.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-embedding-dir", required=True)
    parser.add_argument("--concepts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--top-k", type=int, default=30)
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

    concepts = load_concepts(args.concepts)
    prompt_texts = [c["retrieval_prompt"] for c in concepts]
    model_keys = parse_model_keys(args.models)
    top_k = min(args.top_k, len(ids))

    scores_by_model: Dict[str, np.ndarray] = {}
    ranks_by_model: Dict[str, np.ndarray] = {}
    top_by_model: Dict[str, List[List[int]]] = {}

    for model_key in model_keys:
        img = np.load(emb_dir / f"{model_key}.npy").astype(np.float32)
        txt = embed_prompts(model_key, prompt_texts, args.device, args.local_files_only, args.no_fp16)
        if img.shape[1] != txt.shape[1]:
            raise ValueError(f"dimension mismatch for {model_key}: image {img.shape}, text {txt.shape}")
        sims = txt @ img.T
        scores_by_model[model_key] = sims.astype(np.float32, copy=False)
        rank_rows = []
        top_rows = []
        for p_idx in range(len(concepts)):
            top, ranks, _order = ranked_hits(sims[p_idx], ids, top_k)
            top_rows.append(top)
            rank_rows.append(ranks)
        ranks_by_model[model_key] = np.stack(rank_rows, axis=0)
        top_by_model[model_key] = top_rows
        print(f"computed inference retrievals for {model_key}")

    families = family_stage_keys(model_keys)
    output: Dict[str, object] = {"concepts": concepts, "models": {}, "summary": {}}

    for model_key in model_keys:
        cfg = IMAGE_MODELS[model_key]
        family = cfg["family"]
        baseline_key = families.get(family, {}).get("baseline")
        model_out: Dict[str, List[Dict[str, object]]] = {}
        for p_idx, concept in enumerate(concepts):
            hits = []
            baseline_top = set(top_by_model[baseline_key][p_idx]) if baseline_key else set()
            for rank, idx in enumerate(top_by_model[model_key][p_idx], start=1):
                score = float(scores_by_model[model_key][p_idx, idx])
                item: Dict[str, object] = {
                    "index": int(idx),
                    "image_id": ids[int(idx)],
                    "rank": int(rank),
                    "similarity": round_float(score),
                }
                if baseline_key and model_key != baseline_key:
                    br = int(ranks_by_model[baseline_key][p_idx, idx])
                    bs = float(scores_by_model[baseline_key][p_idx, idx])
                    item.update({
                        "baseline_rank": br,
                        "baseline_similarity": round_float(bs),
                        "rank_delta_vs_baseline": int(br - rank),
                        "similarity_delta_vs_baseline": round_float(score - bs),
                        "status_vs_baseline": "shared" if idx in baseline_top else "new",
                    })
                else:
                    item.update({
                        "baseline_rank": int(rank),
                        "baseline_similarity": round_float(score),
                        "rank_delta_vs_baseline": 0,
                        "similarity_delta_vs_baseline": 0.0,
                        "status_vs_baseline": "baseline",
                    })
                hits.append(item)
            model_out[concept["id"]] = hits
        output["models"][model_key] = model_out

    for family, stage_keys in families.items():
        base_key = stage_keys.get("baseline")
        if not base_key:
            continue
        output["summary"].setdefault(family, {})
        for model_key in stage_keys.values():
            model_summary = {}
            for p_idx, concept in enumerate(concepts):
                base_top = set(top_by_model[base_key][p_idx])
                model_top = set(top_by_model[model_key][p_idx])
                overlap = len(base_top & model_top)
                model_summary[concept["id"]] = {
                    "top_k": top_k,
                    "overlap_with_baseline": overlap,
                    "new_vs_baseline": top_k - overlap,
                }
            output["summary"][family][model_key] = model_summary

    write_json(args.output, output)
    print(f"wrote inference retrievals to {args.output}")


if __name__ == "__main__":
    main()
