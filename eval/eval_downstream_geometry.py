#!/usr/bin/env python3
"""Frozen patch-token linear probes for NYUv2/NAVI depth and normals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import (  # noqa: E402
    assert_patch_grid,
    load_model_bundle,
    patch_features,
    patch_grid,
    processor_geometry_summary,
    resize_like_processor,
    safe_model_name,
    seed_everything,
    transform_rgb_vector_to_grid,
    write_json,
)


@dataclass
class GeometryMetrics:
    benchmark: str
    dataset: str
    task: str
    split: str
    model_name: str
    model_id: str
    processor_id: str
    family: str
    train_samples: int
    eval_samples: int
    epochs: int
    seed: int
    image_size: int
    patch_size: int
    patch_grid: int
    hidden_size: int
    abs_rel: float | None = None
    rmse: float | None = None
    mean_angle: float | None = None
    median_angle: float | None = None
    pct_within_11_25: float | None = None
    pct_within_22_5: float | None = None
    pct_within_30: float | None = None
    processor_geometry: dict | None = None


def depth_image_to_grid(image: Image.Image, processor, grid: int, scale: float | None) -> torch.Tensor:
    target = resize_like_processor(image, processor, Image.Resampling.BILINEAR)
    target = target.resize((grid, grid), Image.Resampling.BILINEAR)
    arr = np.array(target, dtype=np.float32)
    if scale is None:
        scale = 1000.0 if float(np.nanmax(arr)) > 100.0 else 1.0
    arr = arr / float(scale)
    return torch.from_numpy(arr).unsqueeze(0)


def depth_image_to_normals_grid(image: Image.Image, processor, grid: int, scale: float | None) -> torch.Tensor:
    depth = depth_image_to_grid(image, processor, grid, scale).squeeze(0).numpy()
    dzdy, dzdx = np.gradient(depth)
    normals = np.dstack([-dzdx, -dzdy, np.ones_like(depth, dtype=np.float32)])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-6)
    return torch.from_numpy(normals).permute(2, 0, 1).float()


class NYUGeometryDataset(Dataset):
    def __init__(
        self,
        root: Path,
        nyu_repo: Path,
        task: str,
        train: bool,
        processor,
        grid: int,
        max_samples: int | None,
        download: bool,
    ):
        split = "train" if train else "test"
        rgb_dir = root / f"{split}_rgb"
        if task == "both":
            depth_dir = root / f"{split}_depth"
            normals_dir = root / f"{split}_sn"
            if not (rgb_dir.exists() and depth_dir.exists() and normals_dir.exists()):
                raise FileNotFoundError(f"Both-task NYUv2 eval needs extracted {split}_rgb/{split}_depth/{split}_sn under {root}")
            files = sorted(
                path.name
                for path in rgb_dir.glob("*.png")
                if (depth_dir / path.name).exists() and (normals_dir / path.name).exists()
            )
        else:
            target_dir = root / f"{split}_{'depth' if task == 'depth' else 'sn'}"
            if rgb_dir.exists() and target_dir.exists():
                files = sorted(path.name for path in rgb_dir.glob("*.png") if (target_dir / path.name).exists())
            else:
                if str(nyu_repo) not in sys.path:
                    sys.path.insert(0, str(nyu_repo))
                from nyuv2 import NYUv2

                rgb_t = lambda x: x
                sn_t = (lambda x: x) if task == "normals" else None
                depth_t = (lambda x: x) if task == "depth" else None
                ds = NYUv2(
                    root=str(root),
                    train=train,
                    download=download,
                    rgb_transform=rgb_t,
                    sn_transform=sn_t,
                    depth_transform=depth_t,
                )
                files = list(ds._files)
        if max_samples is not None:
            files = files[:max_samples]
        self.task = task
        self.processor = processor
        self.grid = grid
        self.rgb_dir = rgb_dir
        self.depth_dir = root / f"{split}_depth"
        self.normals_dir = root / f"{split}_sn"
        self.target_dir = None if task == "both" else root / f"{split}_{'depth' if task == 'depth' else 'sn'}"
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        name = self.files[idx]
        image = Image.open(self.rgb_dir / name).convert("RGB")
        if self.task == "both":
            target = {
                "depth": depth_image_to_grid(Image.open(self.depth_dir / name), self.processor, self.grid, 1e4),
                "normals": transform_rgb_vector_to_grid(Image.open(self.normals_dir / name), self.processor, self.grid),
            }
        elif self.task == "depth":
            target = depth_image_to_grid(Image.open(self.target_dir / name), self.processor, self.grid, 1e4)
        else:
            target = transform_rgb_vector_to_grid(Image.open(self.target_dir / name), self.processor, self.grid)
        return image, target


def iter_navi_items(root: Path):
    for ann_path in sorted(root.glob("*/*/annotations.json")):
        scene_dir = ann_path.parent
        image_dir = scene_dir / "images"
        depth_dir = scene_dir / "depth"
        mask_dir = scene_dir / "masks"
        if not image_dir.exists() or not depth_dir.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as handle:
            anns = json.load(handle)
        for ann in anns:
            name = ann.get("filename")
            if not name:
                continue
            image_path = image_dir / name
            depth_path = depth_dir / (Path(name).stem + ".png")
            mask_path = mask_dir / (Path(name).stem + ".png")
            if image_path.exists() and depth_path.exists():
                yield {
                    "split": ann.get("split", "train"),
                    "image": image_path,
                    "depth": depth_path,
                    "mask": mask_path if mask_path.exists() else None,
                }


class NAVIGeometryDataset(Dataset):
    def __init__(self, root: Path, task: str, split: str, processor, grid: int, max_samples: int | None):
        split_key = "val" if split == "val" else "train"
        samples = [item for item in iter_navi_items(root) if item["split"] == split_key]
        if max_samples is not None:
            samples = samples[:max_samples]
        self.task = task
        self.processor = processor
        self.grid = grid
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        depth = Image.open(sample["depth"])
        if self.task == "both":
            target = {
                "depth": depth_image_to_grid(depth, self.processor, self.grid, None),
                "normals": depth_image_to_normals_grid(depth, self.processor, self.grid, None),
            }
        elif self.task == "depth":
            target = depth_image_to_grid(depth, self.processor, self.grid, None)
        else:
            target = depth_image_to_normals_grid(depth, self.processor, self.grid, None)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen geometry probe")
    parser.add_argument("--dataset", choices=["nyuv2", "navi"], required=True)
    parser.add_argument("--task", choices=["depth", "normals", "both"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nyu-repo", type=Path, default=Path("/home/localadmin/bz/ReasonCLIP/rebuttal/downstream_repos/pytorch-nyuv2"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results" / "downstream")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-features", action="store_true")
    return parser.parse_args()


def collate(batch):
    images, targets = zip(*batch)
    if isinstance(targets[0], dict):
        stacked = {key: torch.stack([target[key] for target in targets], dim=0) for key in targets[0]}
        return list(images), stacked
    return list(images), torch.stack(list(targets), dim=0)


def build_datasets(args, processor, grid: int):
    if args.dataset == "nyuv2":
        train_ds = NYUGeometryDataset(
            args.data_root,
            args.nyu_repo,
            args.task,
            True,
            processor,
            grid,
            args.max_train,
            args.download,
        )
        eval_ds = NYUGeometryDataset(
            args.data_root,
            args.nyu_repo,
            args.task,
            False,
            processor,
            grid,
            args.max_eval,
            args.download,
        )
        split = "test"
    else:
        train_ds = NAVIGeometryDataset(args.data_root, args.task, "train", processor, grid, args.max_train)
        eval_ds = NAVIGeometryDataset(args.data_root, args.task, "val", processor, grid, args.max_eval)
        split = "val"
    return train_ds, eval_ds, split


def train_head(args, model, processor, head, opt, loader, grid: int, out_dim: int, device: str) -> None:
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for images, target in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            target = target.to(device)
            feats = patch_features(model, processor, images, device)
            pred = head(feats).reshape(feats.shape[0], grid, grid, out_dim).permute(0, 3, 1, 2)
            loss = geometry_loss(args.task, pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={float(np.mean(losses)):.6f}")


def geometry_loss(task: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "depth":
        pred = F.softplus(pred)
        valid = target > 1e-4
        if not valid.any():
            raise RuntimeError("Depth batch has no valid pixels")
        return F.l1_loss(pred[valid], target[valid])
    pred = F.normalize(pred, dim=1, eps=1e-6)
    target_norm = F.normalize(target, dim=1, eps=1e-6)
    return 1.0 - (pred * target_norm).sum(dim=1).mean()


def precompute_features(model, processor, loader, device: str, desc: str):
    feats_all = []
    targets_all = None
    for images, target in tqdm(loader, desc=desc):
        feats = patch_features(model, processor, images, device)
        feats_all.append(feats.detach().to(torch.float16).cpu())
        if isinstance(target, dict):
            if targets_all is None:
                targets_all = {key: [] for key in target}
            for key, value in target.items():
                targets_all[key].append(value.cpu())
        else:
            if targets_all is None:
                targets_all = []
            targets_all.append(target.cpu())
    feats_cpu = torch.cat(feats_all, dim=0)
    if isinstance(targets_all, dict):
        return feats_cpu, {key: torch.cat(values, dim=0) for key, values in targets_all.items()}
    return feats_cpu, torch.cat(targets_all, dim=0)


def train_cached_head(args, task: str, head, opt, feats_cpu, target_cpu, grid: int, out_dim: int, device: str) -> None:
    count = feats_cpu.shape[0]
    for epoch in range(args.epochs):
        head.train()
        losses = []
        order = torch.randperm(count)
        for start in tqdm(range(0, count, args.batch_size), desc=f"{task} epoch {epoch + 1}/{args.epochs}"):
            idx = order[start : start + args.batch_size]
            feats = feats_cpu[idx].to(device=device, dtype=torch.float32, non_blocking=True)
            target = target_cpu[idx].to(device=device, non_blocking=True)
            pred = head(feats).reshape(feats.shape[0], grid, grid, out_dim).permute(0, 3, 1, 2)
            loss = geometry_loss(task, pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"task={task} epoch={epoch + 1} loss={float(np.mean(losses)):.6f}")


def eval_depth_head(head, feats_cpu, target_cpu, grid: int, device: str) -> tuple[float, float]:
    head.eval()
    abs_rel_sum = 0.0
    sq_sum = 0.0
    count = 0
    for start in tqdm(range(0, feats_cpu.shape[0], 64), desc="eval depth", leave=False):
        feats = feats_cpu[start : start + 64].to(device=device, dtype=torch.float32, non_blocking=True)
        target = target_cpu[start : start + 64].to(device=device, non_blocking=True)
        with torch.no_grad():
            pred = head(feats).reshape(feats.shape[0], grid, grid, 1).permute(0, 3, 1, 2)
            pred = F.softplus(pred)
        valid = target > 1e-4
        diff = pred[valid] - target[valid]
        abs_rel_sum += (diff.abs() / target[valid].clamp_min(1e-4)).sum().item()
        sq_sum += (diff**2).sum().item()
        count += int(valid.sum().item())
    if count == 0:
        raise RuntimeError("No valid depth pixels were evaluated")
    return abs_rel_sum / count, float(np.sqrt(sq_sum / count))


def eval_normals_head(head, feats_cpu, target_cpu, grid: int, device: str):
    head.eval()
    angles = []
    for start in tqdm(range(0, feats_cpu.shape[0], 64), desc="eval normals", leave=False):
        feats = feats_cpu[start : start + 64].to(device=device, dtype=torch.float32, non_blocking=True)
        target = target_cpu[start : start + 64].to(device=device, non_blocking=True)
        with torch.no_grad():
            pred = head(feats).reshape(feats.shape[0], grid, grid, 3).permute(0, 3, 1, 2)
            pred = F.normalize(pred, dim=1, eps=1e-6)
            target = F.normalize(target, dim=1, eps=1e-6)
            angle = torch.rad2deg(torch.acos((pred * target).sum(dim=1).clamp(-1.0, 1.0)))
        angles.append(angle.detach().cpu().reshape(-1))
    if not angles:
        raise RuntimeError("No normal pixels were evaluated")
    all_angles = torch.cat(angles)
    return (
        float(all_angles.mean().item()),
        float(all_angles.median().item()),
        float((all_angles < 11.25).float().mean().item()),
        float((all_angles < 22.5).float().mean().item()),
        float((all_angles < 30.0).float().mean().item()),
    )


def eval_uncached(args, model, processor, head, loader, grid: int, device: str):
    feats_all = []
    targets_all = []
    for images, target in tqdm(loader, desc=f"cache eval {args.task}"):
        feats_all.append(patch_features(model, processor, images, device).detach().to(torch.float16).cpu())
        targets_all.append(target.cpu())
    feats_cpu = torch.cat(feats_all, dim=0)
    target_cpu = torch.cat(targets_all, dim=0)
    if args.task == "depth":
        return eval_depth_head(head, feats_cpu, target_cpu, grid, device)
    return eval_normals_head(head, feats_cpu, target_cpu, grid, device)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bundle = load_model_bundle(
        args.model_id,
        processor_id=args.processor_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]
    image_size, patch_size, grid, hidden = patch_grid(model)
    out_dim = 1 if args.task == "depth" else 3

    train_ds, eval_ds, split = build_datasets(args, processor, grid)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError(f"Empty split: train={len(train_ds)}, eval={len(eval_ds)}")
    assert_patch_grid(model, processor, [train_ds[0][0]], device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "processor_id": bundle["processor_id"],
                "family": bundle["family"],
                "dataset": args.dataset,
                "task": args.task,
                "device": device,
                "image_size": image_size,
                "patch_size": patch_size,
                "grid": grid,
                "hidden": hidden,
                "train_samples": len(train_ds),
                "eval_samples": len(eval_ds),
                "processor_geometry": processor_geometry_summary(processor),
            },
            sort_keys=True,
        )
    )


    model_name = args.model_name or safe_model_name(args.model_id)
    if args.task == "both":
        if not args.cache_features:
            raise RuntimeError("--task both requires --cache-features so depth and normals share one feature pass")
        train_feats, train_targets = precompute_features(model, processor, train_loader, device, "cache train features")
        eval_feats, eval_targets = precompute_features(model, processor, eval_loader, device, "cache eval features")
        for task_name, task_out_dim in (("depth", 1), ("normals", 3)):
            head = nn.Linear(hidden, task_out_dim).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            train_cached_head(args, task_name, head, opt, train_feats, train_targets[task_name], grid, task_out_dim, device)
            if task_name == "depth":
                abs_rel, rmse = eval_depth_head(head, eval_feats, eval_targets[task_name], grid, device)
                metric_kwargs = {"abs_rel": abs_rel, "rmse": rmse}
            else:
                mean, median, p11, p22, p30 = eval_normals_head(head, eval_feats, eval_targets[task_name], grid, device)
                metric_kwargs = {
                    "mean_angle": mean,
                    "median_angle": median,
                    "pct_within_11_25": p11,
                    "pct_within_22_5": p22,
                    "pct_within_30": p30,
                }
            metrics = GeometryMetrics(
                benchmark="geometry",
                dataset=args.dataset,
                task=task_name,
                split=split,
                model_name=model_name,
                model_id=args.model_id,
                processor_id=bundle["processor_id"],
                family=bundle["family"],
                train_samples=len(train_ds),
                eval_samples=len(eval_ds),
                epochs=args.epochs,
                seed=args.seed,
                image_size=image_size,
                patch_size=patch_size,
                patch_grid=grid,
                hidden_size=hidden,
                processor_geometry=processor_geometry_summary(processor),
                **metric_kwargs,
            )
            out = args.out_dir / "geometry" / f"{args.dataset}_{task_name}_{safe_model_name(model_name)}.json"
            write_json(out, metrics)
            print(json.dumps(asdict(metrics), sort_keys=True))
            print(f"saved={out}")
        return

    head = nn.Linear(hidden, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cache_features:
        train_feats, train_target = precompute_features(model, processor, train_loader, device, "cache train features")
        eval_feats, eval_target = precompute_features(model, processor, eval_loader, device, "cache eval features")
        train_cached_head(args, args.task, head, opt, train_feats, train_target, grid, out_dim, device)
        if args.task == "depth":
            abs_rel, rmse = eval_depth_head(head, eval_feats, eval_target, grid, device)
            metric_kwargs = {"abs_rel": abs_rel, "rmse": rmse}
        else:
            mean, median, p11, p22, p30 = eval_normals_head(head, eval_feats, eval_target, grid, device)
            metric_kwargs = {
                "mean_angle": mean,
                "median_angle": median,
                "pct_within_11_25": p11,
                "pct_within_22_5": p22,
                "pct_within_30": p30,
            }
    else:
        train_head(args, model, processor, head, opt, train_loader, grid, out_dim, device)
        if args.task == "depth":
            abs_rel, rmse = eval_uncached(args, model, processor, head, eval_loader, grid, device)
            metric_kwargs = {"abs_rel": abs_rel, "rmse": rmse}
        else:
            mean, median, p11, p22, p30 = eval_uncached(args, model, processor, head, eval_loader, grid, device)
            metric_kwargs = {
                "mean_angle": mean,
                "median_angle": median,
                "pct_within_11_25": p11,
                "pct_within_22_5": p22,
                "pct_within_30": p30,
            }

    model_name = args.model_name or safe_model_name(args.model_id)
    metrics = GeometryMetrics(
        benchmark="geometry",
        dataset=args.dataset,
        task=args.task,
        split=split,
        model_name=model_name,
        model_id=args.model_id,
        processor_id=bundle["processor_id"],
        family=bundle["family"],
        train_samples=len(train_ds),
        eval_samples=len(eval_ds),
        epochs=args.epochs,
        seed=args.seed,
        image_size=image_size,
        patch_size=patch_size,
        patch_grid=grid,
        hidden_size=hidden,
        processor_geometry=processor_geometry_summary(processor),
        **metric_kwargs,
    )
    out = args.out_dir / "geometry" / f"{args.dataset}_{args.task}_{safe_model_name(model_name)}.json"
    write_json(out, metrics)
    print(json.dumps(asdict(metrics), sort_keys=True))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
