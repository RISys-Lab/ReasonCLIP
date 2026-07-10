#!/usr/bin/env python3
"""Frozen patch-token linear probes for VOC/ADE20K semantic segmentation."""

from __future__ import annotations

import argparse
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
    safe_model_name,
    seed_everything,
    transform_mask_to_grid,
    write_json,
)


@dataclass
class SegmentationMetrics:
    benchmark: str
    dataset: str
    split: str
    model_name: str
    model_id: str
    processor_id: str
    family: str
    train_samples: int
    val_samples: int
    epochs: int
    seed: int
    image_size: int
    patch_size: int
    patch_grid: int
    hidden_size: int
    pixel_accuracy: float
    mean_iou: float
    valid_class_count: int
    processor_geometry: dict


class VOCDataset(Dataset):
    num_classes = 21

    def __init__(self, root: Path, split: str, processor, grid: int, max_samples: int | None):
        voc = root / "VOCdevkit" / "VOC2012"
        split_file = voc / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing VOC split file: {split_file}")
        ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        if max_samples is not None:
            ids = ids[:max_samples]
        self.processor = processor
        self.grid = grid
        self.items = [
            (voc / "JPEGImages" / f"{item_id}.jpg", voc / "SegmentationClass" / f"{item_id}.png")
            for item_id in ids
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        mask = transform_mask_to_grid(Image.open(mask_path), self.processor, self.grid)
        mask[(mask < 0) | (mask > 20)] = 255
        return image, mask


class ADE20KDataset(Dataset):
    num_classes = 150

    def __init__(self, root: Path, split: str, processor, grid: int, max_samples: int | None):
        sub = "training" if split == "train" else "validation"
        ade = root / "ADEChallengeData2016"
        image_dir = ade / "images" / sub
        ann_dir = ade / "annotations" / sub
        if not image_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(f"Missing ADE20K dirs under {ade}")
        images = sorted(image_dir.glob("*.jpg"))
        if max_samples is not None:
            images = images[:max_samples]
        self.processor = processor
        self.grid = grid
        self.items = [(path, ann_dir / f"{path.stem}.png") for path in images]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        raw = transform_mask_to_grid(Image.open(mask_path), self.processor, self.grid)
        mask = raw.clone()
        valid = (raw >= 1) & (raw <= 150)
        mask[valid] = raw[valid] - 1
        mask[~valid] = 255
        return image, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen dense segmentation probe")
    parser.add_argument("--dataset", choices=["voc", "ade20k"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
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
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def collate(batch):
    images, masks = zip(*batch)
    return list(images), torch.stack(list(masks), dim=0)


def build_dataset(name: str, root: Path, split: str, processor, grid: int, max_samples: int | None):
    if name == "voc":
        return VOCDataset(root, split, processor, grid, max_samples)
    if name == "ade20k":
        return ADE20KDataset(root, split, processor, grid, max_samples)
    raise ValueError(name)


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> None:
    valid = target != 255
    if not valid.any():
        return
    pred = pred[valid].reshape(-1)
    target = target[valid].reshape(-1)
    idx = target * num_classes + pred
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).cpu()


def summarize_confusion(confusion: torch.Tensor) -> tuple[float, float, int]:
    diag = torch.diag(confusion).float()
    total = confusion.sum().float()
    if total <= 0:
        raise RuntimeError("No valid segmentation pixels were evaluated")
    pixel_acc = (diag.sum() / total).item()
    denom = confusion.sum(1).float() + confusion.sum(0).float() - diag
    valid = denom > 0
    mean_iou = (diag[valid] / denom[valid]).mean().item()
    return pixel_acc, mean_iou, int(valid.sum().item())


def evaluate(model, processor, head, loader, num_classes: int, grid: int, device: str) -> tuple[float, float, int]:
    head.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="eval", leave=False):
            masks = masks.to(device)
            feats = patch_features(model, processor, images, device)
            logits = head(feats).reshape(feats.shape[0], grid, grid, num_classes).permute(0, 3, 1, 2)
            pred = logits.argmax(dim=1)
            update_confusion(confusion, pred.cpu(), masks.cpu(), num_classes)
    return summarize_confusion(confusion)


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

    train_ds = build_dataset(args.dataset, args.data_root, "train", processor, grid, args.max_train)
    val_ds = build_dataset(args.dataset, args.data_root, "val", processor, grid, args.max_val)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty split: train={len(train_ds)}, val={len(val_ds)}")
    assert_patch_grid(model, processor, [train_ds[0][0]], device)

    num_classes = train_ds.num_classes
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    print(
        json_line(
            {
                "model_id": args.model_id,
                "processor_id": bundle["processor_id"],
                "family": bundle["family"],
                "dataset": args.dataset,
                "device": device,
                "image_size": image_size,
                "patch_size": patch_size,
                "grid": grid,
                "hidden": hidden,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "processor_geometry": processor_geometry_summary(processor),
            }
        )
    )

    head = nn.Linear(hidden, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            masks = masks.to(device)
            feats = patch_features(model, processor, images, device)
            logits = head(feats).reshape(feats.shape[0], grid, grid, num_classes).permute(0, 3, 1, 2)
            loss = F.cross_entropy(logits, masks, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={float(np.mean(losses)):.6f}")

    pixel_acc, mean_iou, valid_classes = evaluate(model, processor, head, val_loader, num_classes, grid, device)
    model_name = args.model_name or safe_model_name(args.model_id)
    metrics = SegmentationMetrics(
        benchmark="segmentation",
        dataset=args.dataset,
        split="val",
        model_name=model_name,
        model_id=args.model_id,
        processor_id=bundle["processor_id"],
        family=bundle["family"],
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        epochs=args.epochs,
        seed=args.seed,
        image_size=image_size,
        patch_size=patch_size,
        patch_grid=grid,
        hidden_size=hidden,
        pixel_accuracy=pixel_acc,
        mean_iou=mean_iou,
        valid_class_count=valid_classes,
        processor_geometry=processor_geometry_summary(processor),
    )
    out = args.out_dir / "segmentation" / f"{args.dataset}_{safe_model_name(model_name)}.json"
    write_json(out, metrics)
    print(json_line(asdict(metrics)))
    print(f"saved={out}")


def json_line(payload: dict) -> str:
    import json

    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    main()
