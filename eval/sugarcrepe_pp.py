import argparse
import os
import io

import torch
import numpy as np
from datasets import load_dataset, get_dataset_config_names
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoProcessor, SiglipModel, SiglipProcessor


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


def collate_sugarcrepe_pp(batch, processor):
    images, texts_3, filenames = zip(*batch)

    image_inputs = processor(images=list(images), return_tensors="pt")

    # Flatten texts: [B, 3] -> [3B]
    flat_texts = []
    for t3 in texts_3:
        flat_texts.extend(list(t3))

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
    if config_name is None:
        return load_dataset(dataset_name, split=split)
    # In HF datasets, config name is passed as the second positional argument
    return load_dataset(dataset_name, config_name, split=split)


def _resolve_cached_subsets(dataset_name: str, split: str) -> list[str]:
    """
    In offline mode, use configs that are actually present in local cache.
    We'll probe known SugarCrepe_pp configs and keep only those that can be loaded.
    """
    # Note: cache can contain a misspelling 'swap_atribute' (as in your error message)
    candidates = [
        "replace_attribute",
        "replace_object",
        "replace_relation",
        "swap_attribute",
        "swap_atribute",
        "swap_object",
    ]
    ok: list[str] = []
    for cfg in candidates:
        try:
            _ = load_dataset(dataset_name, cfg, split=split)
            ok.append(cfg)
        except Exception:
            continue
    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for x in ok:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "auto" or model_type is None:
        mid = model_id.lower()
        if "siglip" in mid:
            model_type = "siglip"
        else:
            model_type = "clip"

    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Image dir: {image_dir}")
    print(f"Batch size: {batch_size}")
    if max_samples is not None:
        print(f"Max samples: {max_samples}")

    if model_type.lower() == "clip":
        model = AutoModel.from_pretrained(model_id)
        if processor_name is None:
            # Keep consistent with retrieval.py defaults
            processor_name = "openai/clip-vit-large-patch14"
        processor = AutoProcessor.from_pretrained(processor_name)
    elif model_type.lower() == "siglip":
        model = SiglipModel.from_pretrained(model_id)
        if processor_name is None:
            processor_name = "google/siglip2-so400m-patch14-384"
        processor = SiglipProcessor.from_pretrained(processor_name)
    else:
        raise ValueError("model_type must be one of: clip, siglip, auto")

    model.to(device).eval()

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
        collate_fn=lambda b: collate_sugarcrepe_pp(b, processor),
    )

    # Metrics:
    # - ITT: Image+Text task. Count argmin(sim(I,P1), sim(I,P2), sim(I,N)) == N
    # - TOT: Text-only task. For each sample:
    #   (1) sim(P1,P2) > sim(P1,N) => hit
    #   (2) sim(P2,P1) > sim(P2,N) => hit
    #   TOT accuracy = hits / (2 * num_samples)
    total = 0
    itt_correct = 0
    tot_hits = 0

    # Optional: keep running mean of similarities (debug)
    sims_sum_itt = torch.zeros(3, dtype=torch.float64)
    sims_sum_tot = torch.zeros(3, dtype=torch.float64)  # [P1P2, P1N, P2N]

    with torch.no_grad():
        for image_inputs, text_inputs, filenames in tqdm(dataloader, desc="Evaluating"):
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            B = image_features.shape[0]
            # reshape text features back to [B, 3, D]
            text_features = text_features.view(B, 3, -1)

            # ITT: similarity per sample: [B, 3] where 0=P1, 1=P2, 2=N
            sims = torch.einsum("bd,bkd->bk", image_features, text_features)

            # ITT pred: argmin: 0=P1, 1=P2, 2=N
            pred = torch.argmin(sims, dim=1)
            itt_correct += (pred == 2).sum().item()
            total += B

            sims_sum_itt += sims.double().sum(dim=0).cpu()

            # TOT: text-only similarities
            p1 = text_features[:, 0, :]
            p2 = text_features[:, 1, :]
            n = text_features[:, 2, :]
            sim_p1p2 = (p1 * p2).sum(dim=1)  # [B]
            sim_p1n = (p1 * n).sum(dim=1)    # [B]
            sim_p2n = (p2 * n).sum(dim=1)    # [B]

            # Two comparisons per sample
            tot_hits += (sim_p1p2 > sim_p1n).sum().item()
            tot_hits += (sim_p1p2 > sim_p2n).sum().item()  # sim(P2,P1) == sim(P1,P2)

            sims_sum_tot += torch.stack(
                [sim_p1p2.double(), sim_p1n.double(), sim_p2n.double()],
                dim=1,
            ).sum(dim=0).cpu()

    itt_acc = itt_correct / max(1, total)
    tot_acc = tot_hits / max(1, 2 * total)

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
    print(f"Hits: {tot_hits} / {2 * total}")
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
            "metric": "sim(P1,P2) > sim(P1,N) and sim(P2,P1) > sim(P2,N) (counted as 2 hits/sample)",
            "hits": tot_hits,
            "denom": 2 * total,
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
):
    """
    Evaluate ITT/TOT on each dataset subset (config) and write a single summary txt.
    """
    # Resolve subsets. In offline mode, get_dataset_config_names() can yield a misleading ['default'].
    if os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1":
        subsets = _resolve_cached_subsets(dataset_name, split)
        if not subsets:
            try:
                subsets = get_dataset_config_names(dataset_name)
            except Exception:
                subsets = []
    else:
        subsets = get_dataset_config_names(dataset_name)

    # Drop 'default' if present (SugarCrepe_pp doesn't have a cached default config)
    subsets = [s for s in subsets if s not in (None, "default")]

    if not subsets:
        # Fallback: treat as a single dataset without configs
        subsets = [None]

    all_results = []
    for cfg in subsets:
        print("\n" + "=" * 80)
        print(f"Evaluating subset: {cfg}")
        print("=" * 80)
        res = run_sugarcrepe_pp_eval(
            model_id=model_id,
            model_type=model_type,
            dataset_name=dataset_name,
            split=split,
            image_dir=image_dir,
            batch_size=batch_size,
            device=device,
            max_samples=max_samples,
            # Do NOT save per-subset files; we'll save one consolidated txt at the end.
            results_dir=None,
            processor_name=processor_name,
            config_name=cfg,
            save_json=False,
            save_txt=False,
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

        total_all = sum(int(r.get("total", 0)) for r in all_results)
        itt_correct_all = sum(int(r["ITT"]["correct"]) for r in all_results)
        tot_hits_all = sum(int(r["TOT"]["hits"]) for r in all_results)
        itt_avg = itt_correct_all / max(1, total_all)
        tot_avg = tot_hits_all / max(1, 2 * total_all)

        with open(txt_path, "w", encoding="utf-8") as f:
            # First: two task averages
            f.write(f"ITT\t{itt_avg * 100.0:.4f}\n")
            f.write(f"TOT\t{tot_avg * 100.0:.4f}\n")
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

    parser.add_argument("--model_path", type=str, default="openai/clip-vit-large-patch14", help="HF model id or local path")
    parser.add_argument("--model_name", type=str, default="clip", choices=["clip", "siglip", "auto"], help="Model type")

    parser.add_argument("--dataset_name", type=str, default="Aman-J/SugarCrepe_pp", help="HF dataset id")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (SugarCrepe_pp typically uses 'train')")

    parser.add_argument("--image_dir", type=str, default="/home/muzammal/Projects/CLIP-R/data/val2017", help="Local folder containing images, joined with filename")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0/cpu; default auto")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debugging")

    parser.add_argument(
        "--processor_name",
        type=str,
        default=None,
        help="Optional processor name (defaults: CLIP->openai/clip-vit-large-patch14-336, SigLIP->google/siglip2-so400m-patch14-384)",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/muzammal/Projects/CLIP-R/eval/results/sugarcrepe_pp",
        help="Directory to save json result (set to empty to disable saving)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results_dir = args.results_dir
    if results_dir is not None and str(results_dir).strip() == "":
        results_dir = None

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
    )

