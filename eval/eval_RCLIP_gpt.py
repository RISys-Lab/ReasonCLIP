#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate RCLIP 5-way sets with GPT by direct selection.

Input JSONL format per sample:
{
  "id": "...",
  "image_path": "...",
  "sets": [
    {"tag": "...", "gt": "...", "neg": ["...","...","...","..."]},
    ...
  ]
}

For each set, this script sends:
  - image
  - 5 candidates (index 0 is GT)
to GPT, asks it to output ONLY JSON: {"choice": <0..4>}

Metrics:
  - set-level accuracy: predicted index == 0
  - per-tag accuracy
"""

import argparse
import base64
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm


DEFAULT_DATA_BY_VERSION = {
    "v1": "/home/localadmin/bz/RCLIP/rclip_5k_v1_gpt_new.jsonl",
    # "v2": "/home/localadmin/bz/RCLIP/rclip_5k_v2_gpt_new.jsonl",
    "v3": "/home/localadmin/bz/RCLIP/rclip_5k_v3_gpt_new.jsonl",
    "v2_gpt5": "/home/localadmin/bz/RCLIP/rclip_5k_v2_gpt5_new_v2.jsonl",
    "v3_gpt5": "/home/localadmin/bz/RCLIP/rclip_5k_v3_gpt5_new_v2.jsonl",
}

SYSTEM = (
    "You are a strict evaluator for image-text matching. "
    "Given one image and 5 candidate captions, choose the single best-matching caption."
)

USER_TEMPLATE = """Task: choose the single best caption for the image.

Rules:
- Return ONLY JSON, no explanation.
- JSON format must be exactly: {{"choice": <integer 0-4>}}
- Choose exactly one index.

Candidates:
0: {c0}
1: {c1}
2: {c2}
3: {c3}
4: {c4}
"""


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    raise ValueError("line is not a JSON object")
                yield obj
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"


def to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{guess_mime(path)};base64,{b64}"


def extract_sets(sample: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, str, List[str]]]]:
    sid = str(sample.get("id", ""))
    img = sample.get("image_path", "")
    sets = sample.get("sets", [])
    if not isinstance(sid, str) or not sid:
        raise ValueError("missing id")
    if not isinstance(img, str) or not img:
        raise ValueError("missing image_path")
    if not isinstance(sets, list) or not sets:
        raise ValueError("missing sets")

    out: List[Tuple[str, str, List[str]]] = []
    for it in sets:
        tag = it.get("tag", "")
        gt = it.get("gt", "")
        neg = it.get("neg", [])
        if (not isinstance(tag, str) or not tag or
                not isinstance(gt, str) or not gt.strip() or
                not isinstance(neg, list) or len(neg) != 4 or
                any((not isinstance(x, str) or not x.strip()) for x in neg)):
            continue
        out.append((tag, gt.strip(), [x.strip() for x in neg]))
    if not out:
        raise ValueError("no valid set items")
    return sid, img, out


def _extract_json(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        return s[i:j + 1].strip()
    return s


def parse_choice(raw: str) -> Optional[int]:
    txt = (raw or "").strip()
    if not txt:
        return None
    # Strict JSON path first
    try:
        obj = json.loads(_extract_json(txt))
        if isinstance(obj, dict):
            for k in ("choice", "index", "answer", "selected"):
                v = obj.get(k)
                if isinstance(v, int) and 0 <= v <= 4:
                    return v
                if isinstance(v, str) and v.isdigit():
                    n = int(v)
                    if 0 <= n <= 4:
                        return n
    except Exception:
        pass

    # Fallback regex
    m = re.search(r"\b([0-4])\b", txt)
    if m:
        return int(m.group(1))
    return None


class QPSLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / max(qps, 1e-9)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.time()
            self.next_time = now + self.interval


def call_choice_once(
    client: OpenAI,
    model: str,
    image_url: str,
    candidates: List[str],
    max_output_tokens: int,
) -> str:
    prompt = USER_TEMPLATE.format(
        c0=candidates[0], c1=candidates[1], c2=candidates[2], c3=candidates[3], c4=candidates[4]
    )
    req: Dict[str, Any] = dict(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": prompt},
            ]},
        ],
        max_output_tokens=max_output_tokens,
    )
    if model.startswith("gpt-5"):
        req["reasoning"] = {"effort": "minimal"}
        req["text"] = {"verbosity": "low"}
    resp = client.responses.create(**req)

    out = getattr(resp, "output_text", None)
    if isinstance(out, str) and out.strip():
        return out.strip()

    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) in ("output_text", "text"):
                t = getattr(c, "text", "")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
    return "\n".join(chunks).strip()


def choose_with_retry(
    client: OpenAI,
    model: str,
    image_url: str,
    candidates: List[str],
    max_output_tokens: int,
    retries: int,
    base_sleep: float,
    limiter: Optional[QPSLimiter],
) -> Tuple[int, str]:
    last_raw = ""
    last_err = ""
    current_max = max_output_tokens
    for attempt in range(retries + 1):
        try:
            if limiter:
                limiter.wait()
            raw = call_choice_once(client, model, image_url, candidates, current_max)
            last_raw = raw
            choice = parse_choice(raw)
            if choice is None:
                raise ValueError(f"cannot parse choice from: {raw[:300]}")
            return choice, raw
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if ("max_output_tokens" in last_err or "incomplete_details" in last_err) and attempt < retries:
                current_max = min(max(current_max + 64, int(current_max * 1.5)), 2000)
            if attempt < retries:
                sleep = base_sleep * (2 ** attempt) * (0.7 + 0.6 * random.random())
                time.sleep(sleep)
    raise RuntimeError(f"choose failed. last_err={last_err}; last_raw_head={last_raw[:300] if last_raw else '<empty>'}")


def _replace_data_version(path: str, version: str) -> str:
    if version not in ("v1", "v2", "v3"):
        raise ValueError(f"Unsupported version replacement target: {version}")
    new_path, n = re.subn(r"v[123]", version, path, count=1)
    if n == 0:
        raise ValueError(f"Cannot infer data version from --data path: {path}")
    return new_path


def resolve_data_path(version: str, data_override: Optional[str]) -> str:
    if version not in ("v1", "v2", "v3", "v2_gpt5", "v3_gpt5"):
        raise ValueError(f"Unsupported version: {version}")
    if data_override:
        if version in ("v1", "v2", "v3"):
            return _replace_data_version(data_override, version)
        return data_override
    return DEFAULT_DATA_BY_VERSION[version]


def build_eval_items(
    data_path: str,
    max_samples: int = 0,
    sample_ratio: float = 1.0,
    sample_seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(sample_seed)
    items: List[Dict[str, Any]] = []
    for idx, sample in enumerate(tqdm(read_jsonl(data_path), desc="indexing"), 1):
        if max_samples and idx > max_samples:
            break
        if sample_ratio < 1.0 and rng.random() > sample_ratio:
            continue
        if "error" in sample and "sets" not in sample:
            continue
        try:
            sid, img, sets = extract_sets(sample)
        except Exception:
            continue
        if not os.path.exists(img):
            continue
        for tag, gt, negs in sets:
            items.append({
                "id": sid,
                "image_path": img,
                "tag": tag,
                "gt": gt,
                "neg": negs,
            })
    return items


def main():
    ap = argparse.ArgumentParser(description="Evaluate RCLIP by asking GPT to choose among GT+4 negatives.")
    ap.add_argument("--data", default="", help="Optional JSONL path override.")
    ap.add_argument("--data-version", default="v2_gpt5", choices=["v1", "v2", "v3", "v2_gpt5", "v3_gpt5"])
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI model, e.g. gpt-5-mini")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--max-output-tokens", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=0, help="Limit number of source samples; 0 means all.")
    ap.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Randomly keep this fraction of source samples in (0,1]. E.g. 0.1 means 10%.",
    )
    ap.add_argument("--sample-seed", type=int, default=42, help="Seed for random sample-ratio selection.")
    ap.add_argument("--save-errors", default="", help="Optional path to save wrong prediction rows in JSONL.")
    ap.add_argument("--results-dir", default="/home/localadmin/bz/CLIP-R/eval/results/rclip_gpt_select")
    ap.add_argument("--shuffle-seed", type=int, default=42, help="Seed for candidate shuffling.")
    args = ap.parse_args()

    if not (0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")

    data_path = resolve_data_path(args.data_version, args.data or None)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key")
    client = OpenAI(api_key=api_key)

    os.makedirs(args.results_dir, exist_ok=True)
    model_name = str(args.model).replace("/", "_").replace(":", "_")
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    txt_path = os.path.join(args.results_dir, f"rclip_gpt_select_{model_name}_{data_name}.txt")

    items = build_eval_items(
        data_path,
        max_samples=args.max_samples,
        sample_ratio=args.sample_ratio,
        sample_seed=args.sample_seed,
    )
    if not items:
        raise RuntimeError("No valid set items found.")

    limiter = QPSLimiter(args.qps) if args.qps > 0 else None
    lock = threading.Lock()
    rng = random.Random(args.shuffle_seed)

    total = 0
    correct = 0
    per_tag_total: Dict[str, int] = {}
    per_tag_correct: Dict[str, int] = {}
    error_rows: List[Dict[str, Any]] = []

    def run_one(it: Dict[str, Any]) -> Dict[str, Any]:
        original = [it["gt"]] + it["neg"]
        indexed = list(enumerate(original))  # (orig_idx, text), GT is orig_idx=0
        rng.shuffle(indexed)
        cands = [x[1] for x in indexed]
        gt_index = next(i for i, x in enumerate(indexed) if x[0] == 0)
        image_url = to_data_url(it["image_path"])
        pred, raw = choose_with_retry(
            client=client,
            model=args.model,
            image_url=image_url,
            candidates=cands,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            base_sleep=args.sleep,
            limiter=limiter,
        )
        return {
            "pred": pred,
            "raw": raw,
            "cands": cands,
            "gt_index": gt_index,
            "order": [x[0] for x in indexed],  # map shuffled position -> original position
        }

    pbar = tqdm(total=len(items), desc="evaluating")
    if args.workers <= 1:
        for it in items:
            try:
                out = run_one(it)
                is_correct = (out["pred"] == out["gt_index"])
                total += 1
                correct += int(is_correct)
                tag = it["tag"]
                per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                per_tag_correct[tag] = per_tag_correct.get(tag, 0) + int(is_correct)
                if (not is_correct) and args.save_errors:
                    error_rows.append({
                        "id": it["id"],
                        "image_path": it["image_path"],
                        "tag": tag,
                        "gt": it["gt"],
                        "neg": it["neg"],
                        "gt_index_after_shuffle": out["gt_index"],
                        "candidate_order_map": out["order"],
                        "candidates_shuffled": out["cands"],
                        "pred_index": out["pred"],
                        "pred_text": out["cands"][out["pred"]] if 0 <= out["pred"] < 5 else "",
                        "raw": out["raw"],
                    })
            except Exception as e:
                if args.save_errors:
                    error_rows.append({
                        "id": it["id"],
                        "image_path": it["image_path"],
                        "tag": it["tag"],
                        "gt": it["gt"],
                        "neg": it["neg"],
                        "error": f"{type(e).__name__}: {e}",
                    })
            pbar.update(1)
            if total:
                pbar.set_postfix({"set_acc": f"{correct/total:.4f}", "sets": total})
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, it): it for it in items}
            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    out = fut.result()
                    is_correct = (out["pred"] == out["gt_index"])
                    with lock:
                        total += 1
                        correct += int(is_correct)
                        tag = it["tag"]
                        per_tag_total[tag] = per_tag_total.get(tag, 0) + 1
                        per_tag_correct[tag] = per_tag_correct.get(tag, 0) + int(is_correct)
                        if (not is_correct) and args.save_errors:
                            error_rows.append({
                                "id": it["id"],
                                "image_path": it["image_path"],
                                "tag": tag,
                                "gt": it["gt"],
                                "neg": it["neg"],
                                "gt_index_after_shuffle": out["gt_index"],
                                "candidate_order_map": out["order"],
                                "candidates_shuffled": out["cands"],
                                "pred_index": out["pred"],
                                "pred_text": out["cands"][out["pred"]] if 0 <= out["pred"] < 5 else "",
                                "raw": out["raw"],
                            })
                except Exception as e:
                    if args.save_errors:
                        with lock:
                            error_rows.append({
                                "id": it["id"],
                                "image_path": it["image_path"],
                                "tag": it["tag"],
                                "gt": it["gt"],
                                "neg": it["neg"],
                                "error": f"{type(e).__name__}: {e}",
                            })
                pbar.update(1)
                if total:
                    pbar.set_postfix({"set_acc": f"{correct/total:.4f}", "sets": total})
    pbar.close()

    overall = (correct / total) if total else 0.0
    print("\n=== Results ===")
    print(f"Model: {args.model}")
    print(f"Data: {data_path}")
    print(f"Total sets: {total}")
    print(f"Correct sets: {correct}")
    print(f"Set-level Acc (GT ranked #1 among 5): {overall:.6f}")
    print("\n--- Per-tag Acc ---")
    for tag in sorted(per_tag_total.keys()):
        t = per_tag_total[tag]
        c = per_tag_correct.get(tag, 0)
        print(f"{tag:>10s}: {c}/{t} = {c/t:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Results ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Dataset Version: {args.data_version}\n")
        f.write(f"Workers: {args.workers}\n")
        f.write(f"QPS: {args.qps}\n")
        f.write(f"Retries: {args.retries}\n")
        f.write(f"Max Output Tokens: {args.max_output_tokens}\n")
        f.write(f"Max Samples: {args.max_samples}\n")
        f.write(f"Sample Ratio: {args.sample_ratio}\n")
        f.write(f"Sample Seed: {args.sample_seed}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total sets: {total}\n")
        f.write(f"Correct sets: {correct}\n")
        f.write(f"Set-level Acc (GT ranked #1 among 5): {overall:.6f}\n")
        f.write("\n--- Per-tag Acc ---\n")
        for tag in sorted(per_tag_total.keys()):
            t = per_tag_total[tag]
            c = per_tag_correct.get(tag, 0)
            f.write(f"{tag:>10s}: {c}/{t} = {c/t:.6f}\n")
    print(f"\nSaved txt results to: {txt_path}")

    if args.save_errors:
        write_jsonl(args.save_errors, error_rows)
        print(f"Saved error rows to: {args.save_errors} (rows={len(error_rows)})")


if __name__ == "__main__":
    main()

