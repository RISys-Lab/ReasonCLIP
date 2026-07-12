#!/usr/bin/env python3
"""TIPS/Probe3D frozen-backbone probes for NYUv2 and NAVI geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import safe_model_name, seed_everything, write_json  # noqa: E402
from official_geometry_data import (  # noqa: E402
    NAVIProbeDataset,
    NYUDepthDataset,
    NYUGeoNetDataset,
    NYUTestDataset,
    dataset_protocol,
)
from official_probe_utils import (  # noqa: E402
    DeterministicAugmentDataset,
    DeterministicBatchSampler,
    FrozenVisionTower,
)
from tips_dpt import DepthDecoder, NormalsDecoder  # noqa: E402

DEFAULT_DATA_ROOT = REPO_ROOT / "rebuttal" / "downstream_data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "official_downstream"
SIGLIP_SO400M_384_MODEL_ID = "google/siglip-so400m-patch14-384"
SIGLIP_SO400M_384_PAPER_RMSE = {
    ("nyuv2", "depth"): 0.563,
    ("navi", "depth"): 0.069,
    ("nyuv2", "normals"): 24.1,
    ("navi", "normals"): 25.4,
}


@dataclass
class GeometryMetrics:
    rmse: float
    abs_rel: float | None = None
    delta_1: float | None = None
    delta_2: float | None = None
    delta_3: float | None = None


def paper_reference(model_id: str, dataset: str, task: str) -> dict[str, Any] | None:
    if model_id != SIGLIP_SO400M_384_MODEL_ID:
        return None
    return {
        "metric": "depth_rmse" if task == "depth" else "angular_rmse_degrees",
        "value": SIGLIP_SO400M_384_PAPER_RMSE[(dataset, task)],
        "source": "SigLIP 2 Table 2 / TIPS dense protocol",
        "lower_is_better": True,
    }


class LinearDepthHead(nn.Module):
    """DINOv2/TIPS NYUv2 linear 256-bin depth classifier."""

    def __init__(
        self,
        in_channels: int,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        num_bins: int = 256,
        bin_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_bins, kernel_size=1)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_epsilon = bin_epsilon
        self.register_buffer(
            "bin_centers",
            torch.linspace(min_depth, max_depth, num_bins),
        )

    def forward(self, features: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        features = F.interpolate(
            features,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        logits = F.relu(self.classifier(features)) + self.bin_epsilon
        probabilities = logits / logits.sum(dim=1, keepdim=True)
        depth = torch.einsum("bchw,c->bhw", probabilities, self.bin_centers)
        depth = depth.unsqueeze(1)
        return F.interpolate(depth, image_size, mode="bilinear", align_corners=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["nyuv2", "navi"], required=True)
    parser.add_argument("--task", choices=["depth", "normals"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-id")
    parser.add_argument("--model-name")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--nyu-depth-root", type=Path)
    parser.add_argument("--nyu-train-root", type=Path)
    parser.add_argument("--nyu-test-pickle", type=Path)
    parser.add_argument("--navi-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--min-lr-factor", type=float)
    parser.add_argument("--warmup-start-factor", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--sig-loss-weight", type=float)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.5)
    parser.add_argument("--sigloss-warmup-steps", type=int)
    parser.add_argument("--linear-bin-epsilon", type=float, default=0.1)
    parser.add_argument("--dpt-output-activation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scenic-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_protocol_defaults(args: argparse.Namespace) -> None:
    linear_depth = args.dataset == "nyuv2" and args.task == "depth"
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if linear_depth else 5e-4
    if args.warmup_steps is None:
        args.warmup_steps = 12_800 if linear_depth else int(0.15 * args.steps)
    if args.min_lr_factor is None:
        args.min_lr_factor = 1e-8 if linear_depth else 0.01
    if args.warmup_start_factor is None:
        args.warmup_start_factor = 0.001 if linear_depth else 0.01
    if args.grad_clip_norm is None and linear_depth:
        args.grad_clip_norm = 35.0
    if args.sig_loss_weight is None:
        args.sig_loss_weight = 1.0 if linear_depth else 10.0
    if args.sigloss_warmup_steps is None:
        args.sigloss_warmup_steps = 100 if linear_depth else 0
    if args.flip_test is None:
        args.flip_test = linear_depth

    if args.smoke:
        args.steps = min(args.steps, 2)
        args.batch_size = min(args.batch_size, 2)
        args.backbone_batch_size = min(args.backbone_batch_size, 1)
        args.eval_batch_size = min(args.eval_batch_size, 1)
        args.num_workers = 0
        args.max_train = min(args.max_train or 4, 4)
        args.max_eval = min(args.max_eval or 2, 2)
        args.eval_interval = 1
        args.save_interval = 1
        args.log_interval = 1
        args.warmup_steps = min(args.warmup_steps, 1)

    positive = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "backbone_batch_size": args.backbone_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Protocol values must be positive: {invalid}")
    if not 0 <= args.min_lr_factor <= 1:
        raise ValueError("--min-lr-factor must be in [0, 1]")
    if not 0 <= args.warmup_start_factor <= 1:
        raise ValueError("--warmup-start-factor must be in [0, 1]")
    if args.grad_clip_norm is not None and args.grad_clip_norm < 0:
        raise ValueError("--grad-clip-norm must be non-negative")


def build_datasets(args: argparse.Namespace):
    if args.dataset == "nyuv2" and args.task == "depth":
        depth_root = (
            args.nyu_depth_root
            or args.data_root / "NYUv2" / "dinov2_nyu" / "NYU"
        )
        if not depth_root.is_dir():
            raise FileNotFoundError(depth_root)
        train_dataset = NYUDepthDataset(
            depth_root,
            "train",
            augment=args.augment,
            max_samples=args.max_train,
        )
        eval_dataset = NYUDepthDataset(
            depth_root,
            "test",
            augment=False,
            max_samples=args.max_eval,
        )
    elif args.dataset == "nyuv2":
        train_root = args.nyu_train_root or args.data_root / "NYUv2" / "nyuv2_geonet"
        test_pickle = args.nyu_test_pickle or args.data_root / "NYUv2" / "nyuv2_snorm_all.pkl"
        if not train_root.is_dir():
            raise FileNotFoundError(train_root)
        if not test_pickle.is_file():
            raise FileNotFoundError(test_pickle)
        train_dataset = NYUGeoNetDataset(
            train_root,
            task=args.task,
            augment=args.augment,
            max_samples=args.max_train,
        )
        eval_dataset = NYUTestDataset(test_pickle, max_samples=args.max_eval)
    else:
        navi_root = args.navi_root or args.data_root / "NAVI" / "probe3d_navi_v1" / "navi_v1"
        if not navi_root.is_dir():
            raise FileNotFoundError(navi_root)
        train_dataset = NAVIProbeDataset(
            navi_root,
            "trainval",
            augment=args.augment,
            max_samples=args.max_train,
        )
        eval_dataset = NAVIProbeDataset(
            navi_root,
            "test",
            augment=False,
            max_samples=args.max_eval,
        )
    if not train_dataset or not eval_dataset:
        raise RuntimeError("Geometry protocol produced an empty train or evaluation dataset")
    return train_dataset, eval_dataset


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


def extract_final_features(
    tower: FrozenVisionTower,
    images: torch.Tensor,
    batch_size: int,
    torch_dtype: str,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size].to(tower.device_name, non_blocking=True)
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
            chunk = images[start : start + batch_size].to(tower.device_name, non_blocking=True)
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


def build_head(args: argparse.Namespace, tower: FrozenVisionTower) -> nn.Module:
    if args.dataset == "nyuv2" and args.task == "depth":
        head: nn.Module = LinearDepthHead(
            tower.output_channels,
            min_depth=0.001,
            max_depth=10.0,
            bin_epsilon=args.linear_bin_epsilon,
        )
    elif args.task == "depth":
        head = DepthDecoder(
            input_embed_dim=tower.metadata.hidden_size,
            min_depth=0.001,
            max_depth=1.0 if args.dataset == "navi" else 10.0,
            output_activation=args.dpt_output_activation,
        )
    else:
        head = NormalsDecoder(
            input_embed_dim=tower.metadata.hidden_size,
            output_activation=args.dpt_output_activation,
        )
    if args.scenic_init:
        head.apply(scenic_lecun_init)
    return head.to(tower.device_name, dtype=torch.float32)


def learning_rate_at_step(args: argparse.Namespace, step: int) -> float:
    if step <= args.warmup_steps:
        factor = (1 - args.warmup_start_factor) * step / max(1, args.warmup_steps)
        factor += args.warmup_start_factor
    else:
        denominator = max(1, args.steps - args.warmup_steps)
        relative_step = min(1.0, (step - args.warmup_steps) / denominator)
        factor = (1 - args.min_lr_factor) * math.cos(0.5 * math.pi * relative_step)
        factor += args.min_lr_factor
    return args.learning_rate * factor


def one_cycle_beta1_at_step(
    step: int,
    total_steps: int,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
    pct_start: float = 0.3,
) -> float:
    """MMCV OneCycleMomentumUpdaterHook beta1 schedule."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
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
    # Probe3D preserves DINOv2 nominal 1/2/4/6 strides while fixing its
    # missing ellipses so downsampling applies to spatial, not batch/channel, axes.
    prediction_scales = [prediction] + [prediction[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
    target_scales = [target] + [target[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
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


def depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    step: int,
) -> torch.Tensor:
    return sig_loss(
        prediction,
        target,
        args.sig_loss_weight,
        warmup=step < args.sigloss_warmup_steps,
        unbiased_variance=args.dataset == "nyuv2",
    ) + args.gradient_loss_weight * gradient_loss(prediction, target)


def normal_loss(prediction: torch.Tensor, target: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    valid = depth.squeeze(1) > 0
    cosine = F.cosine_similarity(prediction[:, :3], target, dim=1)
    angles = cosine.clamp(min=-1 + 1e-4, max=1 - 1e-4).acos()
    if not valid.any():
        raise RuntimeError("Normals batch contains no valid pixels")
    return angles[valid].mean()


def predict(
    tower: FrozenVisionTower,
    head: nn.Module,
    images: torch.Tensor,
    target_size: tuple[int, int],
    args: argparse.Namespace,
) -> torch.Tensor:
    linear_depth = isinstance(head, LinearDepthHead)
    if linear_depth:
        features: Any = extract_final_features(
            tower,
            images,
            args.backbone_batch_size,
            args.torch_dtype,
        )
    else:
        features = extract_dpt_features(
            tower,
            images,
            args.backbone_batch_size,
            args.torch_dtype,
        )
    with autocast_context(tower.device_name, args.torch_dtype):
        output = head(features, image_size=target_size)
    return output.float()


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
    abs_rel = ((difference.abs() / target.clamp_min(1e-9)) * valid).sum(dim=(1, 2)) / valid_count
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
    cosine = F.cosine_similarity(prediction[:, :3], target, dim=1).clamp(-1, 1)
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


def evaluate(
    tower: FrozenVisionTower,
    head: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
) -> GeometryMetrics:
    head.eval()
    values: dict[str, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"]
            target = batch["depth" if args.task == "depth" else "normals"].to(
                tower.device_name,
                non_blocking=True,
            )
            depth = batch["depth"].to(tower.device_name, non_blocking=True)
            prediction = predict(tower, head, images, target.shape[-2:], args)
            if args.flip_test:
                flipped = predict(tower, head, images.flip(-1), target.shape[-2:], args).flip(-1)
                if args.task == "normals":
                    flipped[:, 0].neg_()
                prediction = 0.5 * (prediction + flipped)
            batch_values = (
                depth_metrics(
                    prediction,
                    target,
                    nyu_crop=args.dataset == "nyuv2",
                )
                if args.task == "depth"
                else normal_metrics(prediction, target, depth)
            )
            for name, metric in batch_values.items():
                values.setdefault(name, []).append(metric.detach().cpu())
    means = {name: float(torch.cat(metric).mean()) for name, metric in values.items()}
    return GeometryMetrics(
        rmse=means["rmse"],
        abs_rel=means.get("abs_rel"),
        delta_1=means.get("delta_1"),
        delta_2=means.get("delta_2"),
        delta_3=means.get("delta_3"),
    )


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


def protocol_summary(
    args: argparse.Namespace,
    tower: FrozenVisionTower,
    train_dataset,
    eval_dataset,
) -> dict[str, Any]:
    linear_depth = args.dataset == "nyuv2" and args.task == "depth"
    return {
        "benchmark": f"{args.dataset}_{args.task}",
        "paper_protocol": "TIPS Appendix B / SigLIP 2 Table 2",
        "frozen_backbone": True,
        "probe": "linear_256_uniform_depth_bins" if linear_depth else "TIPS_DPT_4_layers",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "adamw_betas": [0.9, 0.999],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "warmup_start_factor": args.warmup_start_factor,
        "schedule": "linear-warmup then cosine-to-min-factor",
        "min_lr_factor": args.min_lr_factor,
        "adamw_beta1_schedule": (
            {
                "name": "MMCV OneCycleMomentumUpdaterHook",
                "base_momentum": 0.85,
                "max_momentum": 0.95,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
            }
            if linear_depth
            else None
        ),
        "gradient_clip_norm": args.grad_clip_norm,
        "resume_semantics": "absolute-step deterministic shuffle and per-sample augmentation seeds",
        "optimizer_disclosure": (
            "NYUv2 depth uses the released DINOv2 AdamW, LR, warmup, cosine, "
            "momentum, and clipping settings; TIPS overrides training to 50k steps"
            if linear_depth
            else "TIPS does not disclose every optimizer detail; defaults follow Probe3D"
        ),
        "sig_loss_weight": args.sig_loss_weight if args.task == "depth" else None,
        "sig_loss_variance": (
            "unbiased (DINOv2)" if linear_depth else "biased (Probe3D)"
        ) if args.task == "depth" else None,
        "gradient_loss_weight": args.gradient_loss_weight if args.task == "depth" else None,
        "gradient_loss_scales": (
            "spatial strides [1,2,4,6] using the Probe3D correction of "
            "DINOv2 released batch/channel indexing"
            if args.task == "depth"
            else None
        ),
        "sigloss_warmup_steps": args.sigloss_warmup_steps if args.task == "depth" else None,
        "flip_test": args.flip_test,
        "nyuv2_eigen_evaluation_crop": [45, 471, 41, 601] if linear_depth else None,
        "nyuv2_valid_depth_range": [0.001, 10.0] if linear_depth else None,
        "nyuv2_depth_data_variant": (
            "BTS synchronized frames / released DINOv2 protocol (not Probe3D)"
            if linear_depth
            else None
        ),
        "paper_nominal_train_resolution": [480, 640] if linear_depth else None,
        "released_pipeline_train_resolution": [416, 544] if linear_depth else None,
        "released_dinov2_reference_steps": 38_400 if linear_depth else None,
        "augment": args.augment,
        "scenic_lecun_initialization": args.scenic_init,
        "dpt_output_activation": args.dpt_output_activation if not linear_depth else None,
        "train_dataset": dataset_protocol(train_dataset),
        "eval_dataset": dataset_protocol(eval_dataset),
        "vision_tower": tower.protocol_summary(),
    }


def main() -> None:
    args = parse_args()
    resolve_protocol_defaults(args)
    seed_everything(args.seed)
    train_dataset, eval_dataset = build_datasets(args)
    tower = FrozenVisionTower(
        model_id=args.model_id,
        processor_id=args.processor_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    head = build_head(args, tower)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model_name = args.model_name or safe_model_name(args.model_id)
    run_name = f"{args.dataset}_{args.task}_{model_name}"
    checkpoint_path = args.out_dir / "checkpoints" / f"{run_name}.pt"
    result_path = args.out_dir / f"{run_name}.json"
    best_path = args.out_dir / "checkpoints" / f"{run_name}.best.pt"
    protocol = protocol_summary(args, tower, train_dataset, eval_dataset)
    reference = paper_reference(tower.metadata.model_id, args.dataset, args.task)
    print(json.dumps(protocol, indent=2, sort_keys=True))

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    start_step = 0
    best_rmse = float("inf")
    history: list[dict[str, Any]] = []
    if (args.resume or args.evaluate_only) and checkpoint_path.is_file():
        start_step, best_rmse, history = restore_checkpoint(checkpoint_path, head, optimizer, protocol)
        print(f"[resume] step={start_step} checkpoint={checkpoint_path}")
    if start_step > args.steps:
        raise RuntimeError(f"Checkpoint step {start_step} exceeds configured total {args.steps}")

    if args.evaluate_only:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        metrics = evaluate(tower, head, eval_loader, args)
        payload = {
            "model_name": model_name,
            "model_id": args.model_id,
            "processor_id": args.processor_id or args.model_id,
            "step": start_step,
            "metrics": asdict(metrics),
            "paper_reference": reference,
            "protocol": protocol,
            "history": history,
        }
        write_json(result_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    train_stream = DeterministicAugmentDataset(train_dataset)
    train_sampler = DeterministicBatchSampler(
        len(train_dataset),
        args.batch_size,
        args.seed,
        start_batch=start_step,
        num_batches=args.steps - start_step,
        with_sample_seed=True,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_stream,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    batches = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    interval_started = time.monotonic()
    final_metrics: GeometryMetrics | None = None
    for step in range(start_step, args.steps):
        batch = next(batches)

        learning_rate = learning_rate_at_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            if args.dataset == "nyuv2" and args.task == "depth":
                group["betas"] = (
                    one_cycle_beta1_at_step(step, args.steps),
                    group["betas"][1],
                )
        images = batch["image"]
        depth = batch["depth"].to(tower.device_name, non_blocking=True)
        target = batch["depth" if args.task == "depth" else "normals"].to(
            tower.device_name,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(tower, head, images, target.shape[-2:], args)
        loss = (
            depth_loss(prediction, target, args, step)
            if args.task == "depth"
            else normal_loss(prediction, target, depth)
        )
        loss.backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip_norm)
        optimizer.step()

        running_loss += float(loss.detach())
        running_count += 1
        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == args.steps:
            elapsed = time.monotonic() - interval_started
            print(
                f"[train] step={completed_step}/{args.steps} "
                f"loss={running_loss / running_count:.6f} "
                f"lr={learning_rate:.8g} time={elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0
            interval_started = time.monotonic()

        should_evaluate = completed_step % args.eval_interval == 0 or completed_step == args.steps
        if should_evaluate:
            final_metrics = evaluate(tower, head, eval_loader, args)
            record = {"step": completed_step, **asdict(final_metrics)}
            history.append(record)
            print(f"[eval] {json.dumps(record, sort_keys=True)}", flush=True)
            if final_metrics.rmse < best_rmse:
                best_rmse = final_metrics.rmse
                atomic_torch_save(
                    checkpoint_payload(head, optimizer, completed_step, best_rmse, history, protocol),
                    best_path,
                )
            head.train()

        if completed_step % args.save_interval == 0 or completed_step == args.steps:
            atomic_torch_save(
                checkpoint_payload(head, optimizer, completed_step, best_rmse, history, protocol),
                checkpoint_path,
            )

    if final_metrics is None:
        final_metrics = evaluate(tower, head, eval_loader, args)
        history.append({"step": args.steps, **asdict(final_metrics)})
    payload = {
        "model_name": model_name,
        "model_id": args.model_id,
        "processor_id": args.processor_id or args.model_id,
        "step": args.steps,
        "metrics": asdict(final_metrics),
        "best_rmse": best_rmse,
        "paper_reference": reference,
        "protocol": protocol,
        "history": history,
    }
    write_json(result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
