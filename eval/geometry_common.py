#!/usr/bin/env python3
"""Shared math and runtime helpers for the four geometry evaluations.

This module contains no dataset/task dispatch. Each evaluation entrypoint fixes
its own dataset, head, loss, metric, and protocol constants.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from probe_utils import (  # noqa: E402
    DeterministicAugmentDataset,
    DeterministicBatchSampler,
    FrozenVisionTower,
)

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "downstream_data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "downstream"

PAPER_RMSE = {
    ("openai/clip-vit-large-patch14", "nyuv2", "depth"): (
        "depth_rmse",
        0.553,
        "CLIP L/14@224",
    ),
    ("openai/clip-vit-large-patch14", "navi", "depth"): (
        "depth_rmse",
        0.073,
        "CLIP L/14@224",
    ),
    ("openai/clip-vit-large-patch14", "nyuv2", "normals"): (
        "angular_rmse_degrees",
        24.3,
        "CLIP L/14@224",
    ),
    ("openai/clip-vit-large-patch14", "navi", "normals"): (
        "angular_rmse_degrees",
        25.5,
        "CLIP L/14@224",
    ),
    ("google/siglip-so400m-patch14-384", "nyuv2", "depth"): (
        "depth_rmse",
        0.563,
        "SigLIP 1 So/14@384",
    ),
    ("google/siglip-so400m-patch14-384", "navi", "depth"): (
        "depth_rmse",
        0.069,
        "SigLIP 1 So/14@384",
    ),
    ("google/siglip-so400m-patch14-384", "nyuv2", "normals"): (
        "angular_rmse_degrees",
        24.1,
        "SigLIP 1 So/14@384",
    ),
    ("google/siglip-so400m-patch14-384", "navi", "normals"): (
        "angular_rmse_degrees",
        25.4,
        "SigLIP 1 So/14@384",
    ),
    ("google/siglip2-so400m-patch14-384", "nyuv2", "depth"): (
        "depth_rmse",
        0.466,
        "SigLIP 2 So/14@384",
    ),
    ("google/siglip2-so400m-patch14-384", "navi", "depth"): (
        "depth_rmse",
        0.064,
        "SigLIP 2 So/14@384",
    ),
    ("google/siglip2-so400m-patch14-384", "nyuv2", "normals"): (
        "angular_rmse_degrees",
        23.0,
        "SigLIP 2 So/14@384",
    ),
    ("google/siglip2-so400m-patch14-384", "navi", "normals"): (
        "angular_rmse_degrees",
        25.0,
        "SigLIP 2 So/14@384",
    ),
}


@dataclass
class GeometryMetrics:
    rmse: float
    abs_rel: float | None = None
    delta_1: float | None = None
    delta_2: float | None = None
    delta_3: float | None = None


@dataclass(frozen=True)
class RunLimits:
    steps: int
    batch_size: int
    eval_interval: int
    save_interval: int
    log_interval: int
    max_train: int | None
    max_eval: int | None


def paper_reference(model_id: str, dataset: str, task: str) -> dict[str, Any] | None:
    reference = PAPER_RMSE.get((model_id, dataset, task))
    if reference is None:
        return None
    metric, value, model_label = reference
    return {
        "metric": metric,
        "value": value,
        "source": f"SigLIP 2 Table 2, {model_label} row / TIPS dense protocol",
        "lower_is_better": True,
    }


def geometry_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--backbone-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(seed=42)
    return parser


def run_limits(smoke: bool) -> RunLimits:
    if smoke:
        return RunLimits(
            steps=2,
            batch_size=2,
            eval_interval=1,
            save_interval=1,
            log_interval=1,
            max_train=4,
            max_eval=2,
        )
    return RunLimits(
        steps=50_000,
        batch_size=8,
        eval_interval=10_000,
        save_interval=1_000,
        log_interval=50,
        max_train=None,
        max_eval=None,
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    start_step: int,
    total_steps: int,
    num_workers: int,
) -> DataLoader:
    stream = DeterministicAugmentDataset(dataset)
    sampler = DeterministicBatchSampler(
        len(dataset),
        batch_size,
        seed,
        start_batch=start_step,
        num_batches=total_steps - start_step,
        with_sample_seed=True,
    )
    return DataLoader(
        stream,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluation_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )


def autocast_context(device: str, torch_dtype: str):
    if not device.startswith("cuda"):
        return nullcontext()
    if torch_dtype.lower() in {"bf16", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if torch_dtype.lower() in {"fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def extract_final_features(
    tower: FrozenVisionTower,
    images: torch.Tensor,
    batch_size: int,
    torch_dtype: str,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size].to(
                tower.device_name,
                non_blocking=True,
            )
            chunk = tower.normalize(chunk)
            with autocast_context(tower.device_name, torch_dtype):
                outputs.append(tower.final_features(chunk).detach())
    return torch.cat(outputs, dim=0)


def extract_dpt_features(
    tower: FrozenVisionTower,
    images: torch.Tensor,
    batch_size: int,
    torch_dtype: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    global_outputs: list[list[torch.Tensor]] = [[] for _ in range(4)]
    patch_outputs: list[list[torch.Tensor]] = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size].to(
                tower.device_name,
                non_blocking=True,
            )
            chunk = tower.normalize(chunk)
            with autocast_context(tower.device_name, torch_dtype):
                pairs = tower.dpt_features(chunk)
            for layer, (global_token, patch_map) in enumerate(pairs):
                global_outputs[layer].append(global_token.detach())
                patch_outputs[layer].append(patch_map.detach())
    return [
        (torch.cat(global_outputs[layer], dim=0), torch.cat(patch_outputs[layer], dim=0))
        for layer in range(4)
    ]


def scenic_lecun_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_in = module.in_channels * math.prod(module.kernel_size)
    else:
        return
    target_std = math.sqrt(1.0 / fan_in)
    sampling_std = target_std / 0.8796256610342398
    nn.init.trunc_normal_(
        module.weight,
        mean=0.0,
        std=sampling_std,
        a=-2 * sampling_std,
        b=2 * sampling_std,
    )
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def learning_rate_at_step(
    step: int,
    total_steps: int,
    learning_rate: float,
    warmup_steps: int,
    min_lr_factor: float,
    warmup_start_factor: float,
) -> float:
    if step <= warmup_steps:
        factor = (1 - warmup_start_factor) * step / max(1, warmup_steps)
        factor += warmup_start_factor
    else:
        denominator = max(1, total_steps - warmup_steps)
        relative_step = min(1.0, (step - warmup_steps) / denominator)
        factor = (1 - min_lr_factor) * math.cos(0.5 * math.pi * relative_step)
        factor += min_lr_factor
    return learning_rate * factor


def one_cycle_beta1_at_step(
    step: int,
    total_steps: int,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
    pct_start: float = 0.3,
) -> float:
    """MMCV OneCycleMomentumUpdaterHook beta1 schedule."""

    current = min(max(float(step), 0.0), float(total_steps - 1))
    phase_end = pct_start * total_steps - 1.0
    if current <= phase_end:
        start, end = max_momentum, base_momentum
        progress = current / max(phase_end, 1e-12)
    else:
        start, end = base_momentum, max_momentum
        progress = (current - phase_end) / max(
            total_steps - 1 - phase_end,
            1e-12,
        )
    return end + 0.5 * (start - end) * (
        math.cos(math.pi * progress) + 1.0
    )


def sig_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: float,
    warmup: bool,
    unbiased_variance: bool,
) -> torch.Tensor:
    valid = target > 0
    prediction = prediction[valid]
    target = target[valid]
    if prediction.numel() == 0:
        raise RuntimeError("Depth batch contains no valid pixels")
    difference = torch.log(prediction + 0.001) - torch.log(target + 0.001)
    if warmup:
        return weight * torch.sqrt(0.15 * difference.mean().pow(2) + 1e-12)
    variance = torch.var(difference, unbiased=unbiased_variance)
    return weight * torch.sqrt(variance + 0.15 * difference.mean().pow(2) + 1e-12)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_scales = [prediction] + [
        prediction[..., :: 2 * index, :: 2 * index] for index in range(1, 4)
    ]
    target_scales = [target] + [
        target[..., :: 2 * index, :: 2 * index] for index in range(1, 4)
    ]
    loss = prediction.new_zeros(())
    for prediction_scale, target_scale in zip(prediction_scales, target_scales):
        valid = target_scale > 0
        valid_count = valid.sum().clamp(min=1)
        log_difference = (
            torch.log(prediction_scale + 0.001) - torch.log(target_scale + 0.001)
        ) * valid
        vertical_valid = valid[..., :-2, :] * valid[..., 2:, :]
        horizontal_valid = valid[..., :, :-2] * valid[..., :, 2:]
        vertical = (log_difference[..., :-2, :] - log_difference[..., 2:, :]).abs()
        horizontal = (log_difference[..., :, :-2] - log_difference[..., :, 2:]).abs()
        loss = loss + (
            (vertical * vertical_valid).sum() + (horizontal * horizontal_valid).sum()
        ) / valid_count
    return loss


def normalize_and_resize_normals(
    prediction: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(
            "TIPS NormalsDecoder must return Bx3xHxW predictions, "
            f"got {tuple(prediction.shape)}"
        )
    prediction = F.normalize(prediction.float(), p=2, dim=1)
    prediction = F.interpolate(
        prediction,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )
    return F.normalize(prediction, p=2, dim=1)


def normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError(f"Expected three-channel TIPS normal prediction, got {prediction.shape}")
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError(f"Expected three-channel normal target, got {target.shape}")
    valid = depth.squeeze(1) > 0
    cosine = F.cosine_similarity(prediction, target, dim=1)
    angles = cosine.clamp(min=-1 + 1e-4, max=1 - 1e-4).acos()
    if not valid.any():
        raise RuntimeError("Normals batch contains no valid pixels")
    return angles[valid].mean()


def depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    nyu_crop: bool = False,
) -> dict[str, torch.Tensor]:
    prediction = prediction.squeeze(1)
    target = target.squeeze(1)
    if nyu_crop:
        if prediction.shape[-2:] != (480, 640):
            raise ValueError(
                f"NYUv2 Eigen crop expects 480x640 predictions, got {prediction.shape[-2:]}"
            )
        prediction = prediction[..., 45:471, 41:601]
        target = target[..., 45:471, 41:601]
    valid = (target > 0.001) & (target < 10.0) if nyu_crop else target > 0
    valid_count = valid.sum(dim=(1, 2)).clamp(min=1)
    difference = prediction - target
    rmse = ((difference.pow(2) * valid).sum(dim=(1, 2)) / valid_count).sqrt()
    abs_rel = ((difference.abs() / target.clamp_min(1e-9)) * valid).sum(
        dim=(1, 2)
    ) / valid_count
    ratio = torch.maximum(
        target / prediction.clamp_min(1e-9),
        prediction / target.clamp_min(1e-9),
    )
    return {
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_1": ((ratio < 1.25) * valid).sum(dim=(1, 2)) / valid_count,
        "delta_2": ((ratio < 1.25**2) * valid).sum(dim=(1, 2)) / valid_count,
        "delta_3": ((ratio < 1.25**3) * valid).sum(dim=(1, 2)) / valid_count,
    }


def normal_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cosine = F.cosine_similarity(prediction, target, dim=1).clamp(-1, 1)
    error_degrees = torch.acos(cosine) * (180.0 / math.pi)
    valid = depth.squeeze(1) > 0
    valid_count = valid.sum(dim=(1, 2)).clamp(min=1)
    masked_error = error_degrees * valid
    return {
        "rmse": (masked_error.pow(2).sum(dim=(1, 2)) / valid_count).sqrt(),
        "delta_1": ((error_degrees < 11.25) * valid).sum(dim=(1, 2)) / valid_count,
        "delta_2": ((error_degrees < 22.5) * valid).sum(dim=(1, 2)) / valid_count,
        "delta_3": ((error_degrees < 30.0) * valid).sum(dim=(1, 2)) / valid_count,
    }


def checkpoint_payload(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_rmse: float,
    history: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_rmse": best_rmse,
        "history": history,
        "protocol": protocol,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_protocol: dict[str, Any],
) -> tuple[int, float, list[dict[str, Any]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != expected_protocol:
        raise RuntimeError(f"Geometry checkpoint protocol mismatch: {path}")
    head.load_state_dict(checkpoint["head"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    return (
        int(checkpoint["step"]),
        float(checkpoint.get("best_rmse", float("inf"))),
        list(checkpoint.get("history", [])),
    )
