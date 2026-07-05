#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

from common import IMAGE_MODELS, l2_normalize, read_json, read_jsonl, topk_indices, write_json


def sample_pair_correlation(a: np.ndarray, b: np.ndarray, num_pairs: int, seed: int) -> float:
    n = a.shape[0]
    if n < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    if len(i) < 3:
        return float("nan")
    corr = spearmanr(a[i, j], b[i, j]).correlation
    return float(corr) if corr is not None else float("nan")


def neighbor_overlap(a_nn: np.ndarray, b_nn: np.ndarray) -> float:
    overlaps = []
    for a, b in zip(a_nn, b_nn):
        overlaps.append(len(set(map(int, a)).intersection(map(int, b))) / max(1, len(a)))
    return float(np.mean(overlaps))


def hard_negative_margin(caption_sim: np.ndarray, image_sim: np.ndarray, cap_nn: np.ndarray, percentile: float) -> Tuple[float, float, float]:
    n = caption_sim.shape[0]
    positives = cap_nn[:, 0]
    margins = []
    pos_sims = []
    neg_sims = []
    for i in range(n):
        threshold = np.percentile(caption_sim[i], percentile)
        mask = caption_sim[i] <= threshold
        mask[i] = False
        if not np.any(mask):
            continue
        candidate_idx = np.flatnonzero(mask)
        neg = candidate_idx[np.argmax(image_sim[i, candidate_idx])]
        pos_sim = float(image_sim[i, positives[i]])
        neg_sim = float(image_sim[i, neg])
        margins.append(pos_sim - neg_sim)
        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)
    if not margins:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(margins)), float(np.mean(pos_sims)), float(np.mean(neg_sims))


def model_metrics(caption_sim: np.ndarray, image_emb: np.ndarray, k: int, rsa_pairs: int, seed: int, hard_negative_percentile: float) -> Dict[str, float]:
    image_emb = l2_normalize(image_emb)
    image_sim = image_emb @ image_emb.T
    cap_nn, _ = topk_indices(caption_sim, k=k, exclude_self=True)
    img_nn, _ = topk_indices(image_sim, k=k, exclude_self=True)
    pos_values = np.take_along_axis(image_sim, cap_nn, axis=1)
    margin, pos_sim, hard_neg_sim = hard_negative_margin(caption_sim, image_sim, cap_nn, hard_negative_percentile)
    return {
        f"neighbor_overlap_at_{k}": neighbor_overlap(cap_nn, img_nn),
        f"image_similarity_of_caption_neighbors_at_{k}": float(np.mean(pos_values)),
        "rsa_spearman_sampled": sample_pair_correlation(caption_sim, image_sim, rsa_pairs, seed),
        "triplet_margin_caption_pos_vs_image_hard_neg": margin,
        "triplet_positive_image_similarity": pos_sim,
        "triplet_hard_negative_image_similarity": hard_neg_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute reasoning-space alignment metrics for image embeddings.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--caption-embeddings", required=True)
    parser.add_argument("--image-embedding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rsa-pairs", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard-negative-percentile", type=float, default=25.0)
    args = parser.parse_args()
    rows = read_jsonl(args.annotations)
    ids = [str(r["image_id"]) for r in rows]
    emb_dir = Path(args.image_embedding_dir)
    saved_ids = read_json(emb_dir / "ids.json")
    if list(saved_ids) != ids:
        raise ValueError("annotation ids do not match image embedding ids")
    caption_emb = l2_normalize(np.load(args.caption_embeddings).astype(np.float32))
    if caption_emb.shape[0] != len(rows):
        raise ValueError(f"caption embeddings shape {caption_emb.shape} does not match {len(rows)} rows")
    caption_sim = caption_emb @ caption_emb.T
    result: Dict[str, object] = {"num_rows": len(rows), "k": args.k, "rsa_pairs": args.rsa_pairs, "hard_negative_percentile": args.hard_negative_percentile, "models": {}, "family_deltas": {}}
    for npy in sorted(emb_dir.glob("*.npy")):
        model_key = npy.stem
        if model_key not in IMAGE_MODELS:
            continue
        image_emb = np.load(npy).astype(np.float32)
        metrics = model_metrics(caption_sim, image_emb, args.k, args.rsa_pairs, args.seed, args.hard_negative_percentile)
        result["models"][model_key] = {**IMAGE_MODELS[model_key], **metrics}
        print(model_key, json.dumps(metrics, indent=2))
    by_family: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, metrics in result["models"].items():
        by_family.setdefault(metrics["family"], {})[metrics["stage"]] = metrics
    for family, stages in by_family.items():
        base = stages.get("baseline")
        if not base:
            continue
        deltas = {}
        for stage in ["s1", "s2"]:
            cur = stages.get(stage)
            if not cur:
                continue
            deltas[stage] = {name: float(cur[name] - base[name]) for name in base if isinstance(base.get(name), (int, float)) and isinstance(cur.get(name), (int, float))}
        result["family_deltas"][family] = deltas
    write_json(args.output, result)
    print(f"wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
