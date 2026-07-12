#!/usr/bin/env python3
"""Paper-protocol frozen SigLIP/CLIP semantic-segmentation probe.

The implementation mirrors the TIPS appendix and the released DINOv2
VOC/ADE20K linear-head configs: 512 crops, effective batch 16, 40k updates,
BN plus a 1x1 classifier, AdamW, warmup/poly learning rate, and 512/341
sliding-window validation. The image encoder is always frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import safe_model_name, seed_everything, write_json  # noqa: E402
from official_probe_utils import (  # noqa: E402
    DeterministicAugmentDataset,
    DeterministicBatchSampler,
    FrozenVisionTower,
)

DEFAULT_DATA_ROOT = REPO_ROOT / "rebuttal" / "downstream_data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "official_downstream"
IGNORE_LABEL = 255


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path
    sample_id: str


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    num_classes: int
    train_count: int
    val_count: int
    train_crop: tuple[int, int]
    train_scale_ratio: tuple[float, float]
    test_short_edge: int
    test_max_long_edge: int
    slide_crop: tuple[int, int]
    slide_stride: tuple[int, int]
    uses_voc_sbd_aug: bool
    reduce_zero_label: bool


class BNLinearSegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bn(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["voc", "ade20k"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-id")
    parser.add_argument("--model-name")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--backbone-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--warmup-ratio", type=float, default=1e-6)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_voc_records(data_root: Path) -> tuple[list[SegmentationRecord], list[SegmentationRecord]]:
    voc_root = data_root / "VOCdevkit" / "VOC2012"
    split_root = voc_root / "ImageSets" / "Segmentation"
    train_ids = _read_ids(split_root / "train.txt")
    aug_ids = _read_ids(split_root / "aug.txt")
    val_ids = _read_ids(split_root / "val.txt")

    sbd_image_root = data_root / "VOCdevkit" / "VOCaug" / "benchmark_RELEASE" / "dataset" / "img"
    aug_mask_root = voc_root / "SegmentationClassAug"
    if len(train_ids) != 1464 or len(aug_ids) != 9118 or len(val_ids) != 1449:
        raise RuntimeError(
            "VOC official split mismatch: expected train=1464, aug=9118, val=1449; "
            f"got {len(train_ids)}, {len(aug_ids)}, {len(val_ids)}"
        )

    train_records = []
    for sample_id in train_ids:
        train_records.append(
            SegmentationRecord(
                voc_root / "JPEGImages" / f"{sample_id}.jpg",
                voc_root / "SegmentationClass" / f"{sample_id}.png",
                sample_id,
            )
        )
    for sample_id in aug_ids:
        image_path = voc_root / "JPEGImages" / f"{sample_id}.jpg"
        if not image_path.is_file():
            image_path = sbd_image_root / f"{sample_id}.jpg"
        train_records.append(
            SegmentationRecord(
                image_path,
                aug_mask_root / f"{sample_id}.png",
                sample_id,
            )
        )
    val_records = [
        SegmentationRecord(
            voc_root / "JPEGImages" / f"{sample_id}.jpg",
            voc_root / "SegmentationClass" / f"{sample_id}.png",
            sample_id,
        )
        for sample_id in val_ids
    ]
    return train_records, val_records


def build_ade_records(data_root: Path) -> tuple[list[SegmentationRecord], list[SegmentationRecord]]:
    root = data_root / "ADEChallengeData2016"

    def records(split: str) -> list[SegmentationRecord]:
        image_root = root / "images" / split
        mask_root = root / "annotations" / split
        output = []
        for image_path in sorted(image_root.glob("*.jpg")):
            mask_path = mask_root / f"{image_path.stem}.png"
            output.append(SegmentationRecord(image_path, mask_path, image_path.stem))
        return output

    train_records = records("training")
    val_records = records("validation")
    if len(train_records) != 20_210 or len(val_records) != 2000:
        raise RuntimeError(
            "ADE20K official split mismatch: expected train=20210, val=2000; "
            f"got {len(train_records)}, {len(val_records)}"
        )
    return train_records, val_records


def verify_records(records: list[SegmentationRecord], split: str) -> None:
    missing = [record for record in records if not record.image_path.is_file() or not record.mask_path.is_file()]
    if missing:
        preview = ", ".join(
            f"{record.sample_id} ({record.image_path}, {record.mask_path})" for record in missing[:5]
        )
        raise FileNotFoundError(f"{len(missing)} missing {split} image/mask pairs: {preview}")


def resize_keep_ratio(
    image: np.ndarray,
    mask: np.ndarray | None,
    short_edge: int,
    max_long_edge: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = image.shape[:2]
    # MMSeg first multiplies both members of img_scale by ratio and truncates
    # to int, then calls mmcv.imrescale with that target tuple.
    target_long = int(max_long_edge * ratio)
    target_short = int(short_edge * ratio)
    scale = min(
        target_long / max(height, width),
        target_short / min(height, width),
    )
    new_w = max(1, int(width * scale + 0.5))
    new_h = max(1, int(height * scale + 0.5))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 512,
    cat_max_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    margin_h = max(height - crop_h, 0)
    margin_w = max(width - crop_w, 0)
    top = left = 0
    for _ in range(10):
        top = np.random.randint(0, margin_h + 1) if margin_h else 0
        left = np.random.randint(0, margin_w + 1) if margin_w else 0
        candidate = mask[top : top + crop_h, left : left + crop_w]
        labels, counts = np.unique(candidate, return_counts=True)
        counts = counts[labels != IGNORE_LABEL]
        if len(counts) > 1 and counts.max() / counts.sum() < cat_max_ratio:
            break
    return (
        image[top : top + crop_h, left : left + crop_w],
        mask[top : top + crop_h, left : left + crop_w],
    )


def photometric_distortion(image: np.ndarray) -> np.ndarray:
    """MMSegmentation 0.27 PhotoMetricDistortion for an RGB uint8 image."""

    def convert(value: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        value = value.astype(np.float32) * alpha + beta
        return np.clip(value, 0, 255).astype(np.uint8)

    image = image.copy()
    if np.random.randint(2):
        image = convert(image, beta=float(np.random.uniform(-32.0, 32.0)))

    contrast_first = bool(np.random.randint(2))
    if contrast_first and np.random.randint(2):
        image = convert(image, alpha=float(np.random.uniform(0.5, 1.5)))

    if np.random.randint(2):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = convert(hsv[..., 1], alpha=float(np.random.uniform(0.5, 1.5)))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if np.random.randint(2):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + np.random.randint(-18, 18)) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if not contrast_first and np.random.randint(2):
        image = convert(image, alpha=float(np.random.uniform(0.5, 1.5)))
    return image


def load_rgb_mask(record: SegmentationRecord, reduce_zero_label: bool) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(record.image_path) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    with Image.open(record.mask_path) as mask_file:
        mask = np.asarray(mask_file, dtype=np.uint8).copy()
    if image.shape[:2] != mask.shape:
        raise RuntimeError(
            f"Image/mask shape mismatch for {record.sample_id}: {image.shape[:2]} vs {mask.shape}"
        )
    if reduce_zero_label:
        mask[mask == 0] = IGNORE_LABEL
        mask = mask.astype(np.int16) - 1
        mask[mask == IGNORE_LABEL - 1] = IGNORE_LABEL
        mask = mask.astype(np.uint8)
    return image, mask


class OfficialSegmentationTrainDataset(Dataset):
    def __init__(
        self,
        records: list[SegmentationRecord],
        tower: FrozenVisionTower,
        max_long_edge: int,
        reduce_zero_label: bool,
    ) -> None:
        self.records = records
        self.max_long_edge = max_long_edge
        self.reduce_zero_label = reduce_zero_label
        self.mean = torch.tensor(tower.metadata.image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(tower.metadata.image_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = load_rgb_mask(self.records[index], self.reduce_zero_label)
        ratio = float(np.random.uniform(0.5, 2.0))
        image, mask = resize_keep_ratio(image, mask, 512, self.max_long_edge, ratio)
        image, mask = random_crop(image, mask)
        if np.random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        image = photometric_distortion(image)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()

        padded_image = torch.zeros((3, 512, 512), dtype=torch.float32)
        padded_mask = torch.full((512, 512), IGNORE_LABEL, dtype=torch.long)
        height, width = mask_tensor.shape
        padded_image[:, :height, :width] = image_tensor
        padded_mask[:height, :width] = mask_tensor
        return padded_image, padded_mask


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def autocast_context(device: str, torch_dtype: str):
    if not device.startswith("cuda"):
        return nullcontext()
    if torch_dtype.lower() in {"bf16", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if torch_dtype.lower() in {"fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def extract_feature_batch(
    tower: FrozenVisionTower,
    images: torch.Tensor,
    backbone_batch_size: int,
    torch_dtype: str,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, images.shape[0], backbone_batch_size):
            chunk = images[start : start + backbone_batch_size].to(
                tower.device_name,
                non_blocking=True,
            )
            with autocast_context(tower.device_name, torch_dtype):
                outputs.append(tower.final_features(chunk).detach())
    return torch.cat(outputs, dim=0)


def learning_rate_at_step(args: argparse.Namespace, step: int) -> float:
    regular_factor = max(0.0, 1.0 - step / args.steps)
    if step < args.warmup_steps:
        progress = step / max(1, args.warmup_steps)
        warmup_factor = args.warmup_ratio + (1.0 - args.warmup_ratio) * progress
    else:
        warmup_factor = 1.0
    return args.learning_rate * regular_factor * warmup_factor


def resize_for_test(image: np.ndarray, dataset: str) -> np.ndarray:
    max_long = 2048 if dataset == "voc" else 99_999_999
    resized, _ = resize_keep_ratio(image, None, 512, max_long, 1.0)
    return resized


def slide_starts(length: int, crop: int, stride: int) -> list[int]:
    if length <= crop:
        return [0]
    count = math.ceil((length - crop) / stride) + 1
    return [min(index * stride, length - crop) for index in range(count)]


def infer_sliding(
    tower: FrozenVisionTower,
    head: BNLinearSegmentationHead,
    image: np.ndarray,
    num_classes: int,
    dataset: str,
    torch_dtype: str,
    eval_batch_size: int,
) -> torch.Tensor:
    resized = resize_for_test(image, dataset)
    image_tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(tower.metadata.image_mean).view(3, 1, 1)
    std = torch.tensor(tower.metadata.image_std).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    height, width = image_tensor.shape[-2:]
    y_starts = slide_starts(height, 512, 341)
    x_starts = slide_starts(width, 512, 341)
    windows = [(y, x) for y in y_starts for x in x_starts]

    predictions = torch.zeros((num_classes, height, width), device=tower.device_name, dtype=torch.float32)
    counts = torch.zeros((1, height, width), device=tower.device_name, dtype=torch.float32)
    for batch_start in range(0, len(windows), eval_batch_size):
        locations = windows[batch_start : batch_start + eval_batch_size]
        crop_tensors = []
        crop_shapes = []
        for y, x in locations:
            crop = image_tensor[:, y : min(y + 512, height), x : min(x + 512, width)]
            crop_shapes.append(crop.shape[-2:])
            if crop.shape[-2:] != (512, 512):
                crop = F.pad(crop, (0, 512 - crop.shape[-1], 0, 512 - crop.shape[-2]))
            crop_tensors.append(crop)
        crops = torch.stack(crop_tensors)
        features = extract_feature_batch(tower, crops, eval_batch_size, torch_dtype)
        with torch.no_grad():
            logits = head(features.float())
            logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        logits = logits.float()
        for index, ((y, x), (crop_h, crop_w)) in enumerate(zip(locations, crop_shapes)):
            predictions[:, y : y + crop_h, x : x + crop_w] += logits[index, :, :crop_h, :crop_w]
            counts[:, y : y + crop_h, x : x + crop_w] += 1
    if torch.any(counts == 0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return predictions / counts


def evaluate(
    args: argparse.Namespace,
    tower: FrozenVisionTower,
    head: BNLinearSegmentationHead,
    records: list[SegmentationRecord],
    num_classes: int,
    reduce_zero_label: bool,
) -> dict[str, float | int]:
    head.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    started = time.monotonic()
    for index, record in enumerate(records, start=1):
        image, target = load_rgb_mask(record, reduce_zero_label)
        logits = infer_sliding(
            tower,
            head,
            image,
            num_classes,
            args.dataset,
            args.torch_dtype,
            args.eval_batch_size,
        )
        if logits.shape[-2:] != target.shape:
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=target.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        prediction = logits.argmax(dim=0).cpu()
        target_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        valid = target_tensor != IGNORE_LABEL
        encoded = target_tensor[valid] * num_classes + prediction[valid]
        confusion += torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
        if index % 50 == 0 or index == len(records):
            elapsed = time.monotonic() - started
            print(f"[eval] {index}/{len(records)} images ({elapsed:.1f}s)", flush=True)

    true_positive = confusion.diag().double()
    union = confusion.sum(0).double() + confusion.sum(1).double() - true_positive
    valid_classes = union > 0
    class_iou = torch.full((num_classes,), float("nan"), dtype=torch.float64)
    class_iou[valid_classes] = true_positive[valid_classes] / union[valid_classes]
    total = confusion.sum().item()
    metrics: dict[str, float | int] = {
        "miou": float(class_iou[valid_classes].mean().item()),
        "pixel_accuracy": float(true_positive.sum().item() / total),
        "valid_classes": int(valid_classes.sum().item()),
        "valid_pixels": int(total),
        "eval_seconds": float(time.monotonic() - started),
    }
    print(
        f"[eval] mIoU={metrics['miou'] * 100:.2f} "
        f"pixel_acc={metrics['pixel_accuracy'] * 100:.2f}",
        flush=True,
    )
    return metrics


def checkpoint_payload(
    args: argparse.Namespace,
    step: int,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_miou: float,
    history: list[dict],
) -> dict:
    return {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "processor_id": args.processor_id or args.model_id,
        "resume_protocol": "absolute-step-v1",
        "step": step,
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "history": history,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(
    path: Path,
    args: argparse.Namespace,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, list[dict]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = (args.dataset, args.model_id, args.processor_id or args.model_id)
    actual = (checkpoint["dataset"], checkpoint["model_id"], checkpoint["processor_id"])
    if actual != expected:
        raise RuntimeError(f"Checkpoint spec mismatch: expected {expected}, got {actual}")
    checkpoint_step = int(checkpoint["step"])
    resume_protocol = checkpoint.get("resume_protocol")
    if resume_protocol != "absolute-step-v1":
        if not args.evaluate_only and checkpoint_step < args.steps:
            raise RuntimeError(
                "Cannot deterministically continue this legacy segmentation checkpoint; "
                "restart with --no-resume or use --evaluate-only"
            )
        print(
            "[resume] accepting legacy checkpoint for evaluation/completed run; "
            "its incomplete training stream is not resumable",
            flush=True,
        )
    head.load_state_dict(checkpoint["head"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    return (
        checkpoint_step,
        float(checkpoint.get("best_miou", -1.0)),
        list(checkpoint.get("history", [])),
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.effective_batch_size = min(args.effective_batch_size, 2)
        args.backbone_batch_size = min(args.backbone_batch_size, 1)
        args.max_train = args.max_train or 4
        args.max_val = args.max_val or 2
        args.eval_interval = 1
        args.save_interval = 1
    if args.effective_batch_size % args.backbone_batch_size != 0:
        raise ValueError("effective batch size must be divisible by backbone batch size")
    seed_everything(args.seed)

    tower = FrozenVisionTower(
        args.model_id,
        processor_id=args.processor_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    if args.dataset == "voc":
        train_records, val_records = build_voc_records(args.data_root)
        num_classes = 21
        max_long_edge = 2048
        reduce_zero_label = False
        paper_reference = 73.8 if tower.metadata.model_id == "google/siglip-so400m-patch14-384" else None
    else:
        train_records, val_records = build_ade_records(args.data_root)
        num_classes = 150
        max_long_edge = 99_999_999
        reduce_zero_label = True
        paper_reference = 40.8 if tower.metadata.model_id == "google/siglip-so400m-patch14-384" else None
    verify_records(train_records, "train")
    verify_records(val_records, "validation")
    if args.max_train is not None:
        train_records = train_records[: args.max_train]
    if args.max_val is not None:
        val_records = val_records[: args.max_val]

    dataset_protocol = DatasetProtocol(
        name=args.dataset,
        num_classes=num_classes,
        train_count=len(train_records),
        val_count=len(val_records),
        train_crop=(512, 512),
        train_scale_ratio=(0.5, 2.0),
        test_short_edge=512,
        test_max_long_edge=max_long_edge,
        slide_crop=(512, 512),
        slide_stride=(341, 341),
        uses_voc_sbd_aug=args.dataset == "voc",
        reduce_zero_label=reduce_zero_label,
    )
    print(json.dumps({"model": tower.protocol_summary(), "dataset": asdict(dataset_protocol)}, indent=2))

    model_name = args.model_name or safe_model_name(args.model_id)
    result_path = args.out_dir / "segmentation" / f"{args.dataset}_{model_name}.json"
    checkpoint_path = args.out_dir / "checkpoints" / "segmentation" / f"{args.dataset}_{model_name}.pt"
    head = BNLinearSegmentationHead(tower.output_channels, num_classes).to(tower.device_name)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    start_step = 0
    best_miou = -1.0
    history: list[dict] = []
    if args.resume and checkpoint_path.is_file():
        start_step, best_miou, history = restore_checkpoint(checkpoint_path, args, head, optimizer)
        print(f"Resumed {checkpoint_path} at step {start_step}", flush=True)
    if start_step > args.steps:
        raise RuntimeError(f"Checkpoint step {start_step} exceeds configured total {args.steps}")

    if args.evaluate_only:
        if start_step == 0:
            raise RuntimeError("--evaluate-only requires an existing checkpoint")
        metrics = evaluate(args, tower, head, val_records, num_classes, reduce_zero_label)
    else:
        train_dataset = OfficialSegmentationTrainDataset(
            train_records,
            tower,
            max_long_edge,
            reduce_zero_label,
        )
        train_stream = DeterministicAugmentDataset(train_dataset)
        train_sampler = DeterministicBatchSampler(
            len(train_dataset),
            args.effective_batch_size,
            args.seed,
            start_batch=start_step,
            num_batches=args.steps - start_step,
            with_sample_seed=True,
        )
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            train_stream,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
        )
        batches = iter(loader)
        head.train()
        running_loss = 0.0
        running_count = 0
        interval_started = time.monotonic()
        metrics = None
        for step in range(start_step, args.steps):
            lr = learning_rate_at_step(args, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images, targets = next(batches)
            features = extract_feature_batch(
                tower,
                images,
                args.backbone_batch_size,
                args.torch_dtype,
            )
            targets = targets.to(tower.device_name, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.float())
            logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_LABEL)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            running_count += 1
            completed_step = step + 1

            if completed_step % args.log_interval == 0 or completed_step == args.steps:
                elapsed = time.monotonic() - interval_started
                print(
                    f"[train] step={completed_step}/{args.steps} "
                    f"loss={running_loss / running_count:.6f} "
                    f"lr={lr:.8g} time={elapsed:.1f}s",
                    flush=True,
                )
                running_loss = 0.0
                running_count = 0
                interval_started = time.monotonic()

            should_evaluate = completed_step % args.eval_interval == 0 or completed_step == args.steps
            if should_evaluate:
                metrics = evaluate(args, tower, head, val_records, num_classes, reduce_zero_label)
                history.append({"step": completed_step, **metrics})
                best_miou = max(best_miou, float(metrics["miou"]))
                head.train()

            if completed_step % args.save_interval == 0 or completed_step == args.steps:
                save_checkpoint(
                    checkpoint_path,
                    checkpoint_payload(args, completed_step, head, optimizer, best_miou, history),
                )
        if metrics is None:
            metrics = evaluate(args, tower, head, val_records, num_classes, reduce_zero_label)

    result = {
        "benchmark": "official_frozen_backbone_segmentation",
        "dataset": asdict(dataset_protocol),
        "model": tower.protocol_summary(),
        "model_name": model_name,
        "training": {
            "steps": args.steps,
            "effective_batch_size": args.effective_batch_size,
            "backbone_batch_size": args.backbone_batch_size,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "lr_policy": "linear_warmup_then_poly_power_1",
            "seed": args.seed,
            "frozen_backbone": True,
            "head": "BatchNorm2d + Conv2d(1x1)",
            "resume_protocol": "absolute-step-v1",
            "shuffle": "absolute batch index with epoch-specific permutations",
            "augmentation_rng": "per-sample deterministic seed",
        },
        "metrics": metrics,
        "best_miou": best_miou,
        "history": history,
        "paper_reference_miou_percent": paper_reference,
        "paper_reference_source": "SigLIP 2 Table 2 / TIPS dense protocol" if paper_reference else None,
        "checkpoint": str(checkpoint_path),
    }
    write_json(result_path, result)
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
