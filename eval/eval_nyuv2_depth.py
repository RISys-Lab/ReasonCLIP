#!/usr/bin/env python3
"""NYUv2 frozen-backbone depth evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import safe_model_name, seed_everything, write_json  # noqa: E402
from geometry_common import (  # noqa: E402
    GeometryMetrics,
    RunLimits,
    atomic_torch_save,
    autocast_context,
    checkpoint_payload,
    depth_metrics,
    evaluation_loader,
    extract_final_features,
    geometry_parser,
    gradient_loss,
    learning_rate_at_step,
    one_cycle_beta1_at_step,
    paper_reference,
    restore_checkpoint,
    run_limits,
    sig_loss,
    train_loader,
)
from geometry_data import NYUDepthDataset, dataset_protocol  # noqa: E402
from probe_utils import FrozenVisionTower  # noqa: E402

DATASET = "nyuv2"
TASK = "depth"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 12_800
WARMUP_START_FACTOR = 0.001
MIN_LR_FACTOR = 1e-8
GRAD_CLIP_NORM = 35.0
SIG_LOSS_WEIGHT = 1.0
SIGLOSS_WARMUP_STEPS = 100
GRADIENT_LOSS_WEIGHT = 0.5


class LinearDepthHead(nn.Module):
    """TIPS/DINOv2 linear 256-bin depth classifier."""

    def __init__(
        self,
        in_channels: int,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        num_bins: int = 256,
        bin_epsilon: float = 0.1,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_bins, kernel_size=1)
        self.patch_size = patch_size
        self.bin_epsilon = bin_epsilon
        self.register_buffer(
            "bin_centers",
            torch.linspace(min_depth, max_depth, num_bins),
        )

    def forward(self, features: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        padded_size = (
            features.shape[-2] * self.patch_size,
            features.shape[-1] * self.patch_size,
        )
        features = F.interpolate(
            features,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        logits = F.relu(self.classifier(features)) + self.bin_epsilon
        probabilities = logits / logits.sum(dim=1, keepdim=True)
        depth = torch.einsum("bchw,c->bhw", probabilities, self.bin_centers).unsqueeze(1)
        depth = F.interpolate(depth, padded_size, mode="bilinear", align_corners=False)
        top = (padded_size[0] - image_size[0]) // 2
        left = (padded_size[1] - image_size[1]) // 2
        return depth[..., top : top + image_size[0], left : left + image_size[1]]


def parse_args() -> argparse.Namespace:
    return geometry_parser(__doc__).parse_args()


def build_datasets(args: argparse.Namespace, limits: RunLimits):
    root = args.data_root / "NYUv2" / "dinov2_nyu" / "NYU"
    train_dataset = NYUDepthDataset(
        root,
        "train",
        augment=True,
        max_samples=limits.max_train,
    )
    eval_dataset = NYUDepthDataset(
        root,
        "test",
        augment=False,
        max_samples=limits.max_eval,
    )
    return train_dataset, eval_dataset


def predict(
    tower: FrozenVisionTower,
    head: LinearDepthHead,
    images: torch.Tensor,
    target_size: tuple[int, int],
    args: argparse.Namespace,
) -> torch.Tensor:
    features = extract_final_features(
        tower,
        images,
        args.backbone_batch_size,
        args.torch_dtype,
    )
    with autocast_context(tower.device_name, args.torch_dtype):
        prediction = head(features, image_size=target_size)
    return prediction.float()


def loss_at_step(
    prediction: torch.Tensor,
    target: torch.Tensor,
    step: int,
) -> torch.Tensor:
    target = target.masked_fill(target > 10.0, 0)
    return sig_loss(
        prediction,
        target,
        SIG_LOSS_WEIGHT,
        warmup=step < SIGLOSS_WARMUP_STEPS,
        unbiased_variance=True,
    ) + GRADIENT_LOSS_WEIGHT * gradient_loss(prediction, target)


def evaluate(
    tower: FrozenVisionTower,
    head: LinearDepthHead,
    loader: DataLoader,
    args: argparse.Namespace,
) -> GeometryMetrics:
    head.eval()
    values: dict[str, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"]
            target = batch["depth"].to(tower.device_name, non_blocking=True)
            prediction = predict(tower, head, images, target.shape[-2:], args)
            flipped = predict(
                tower,
                head,
                images.flip(-1),
                target.shape[-2:],
                args,
            ).flip(-1)
            for name, metric in depth_metrics(
                0.5 * (prediction + flipped),
                target,
                nyu_crop=True,
            ).items():
                values.setdefault(name, []).append(metric.detach().cpu())
    means = {name: float(torch.cat(metric).mean()) for name, metric in values.items()}
    return GeometryMetrics(
        rmse=means["rmse"],
        abs_rel=means["abs_rel"],
        delta_1=means["delta_1"],
        delta_2=means["delta_2"],
        delta_3=means["delta_3"],
    )


def protocol_summary(
    tower: FrozenVisionTower,
    train_dataset: NYUDepthDataset,
    eval_dataset: NYUDepthDataset,
    limits: RunLimits,
) -> dict[str, Any]:
    return {
        "benchmark": "nyuv2_depth",
        "paper_protocol": "TIPS Appendix B / SigLIP 2 Table 2",
        "frozen_backbone": True,
        "head": "1x1 Conv -> 256 uniform bins in [0.001, 10m]",
        "input_alignment": "center-pad patch-14 input then crop prediction to 480x640",
        "training": {
            "steps": limits.steps,
            "batch_size": limits.batch_size,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "warmup_start_factor": WARMUP_START_FACTOR,
            "min_lr_factor": MIN_LR_FACTOR,
            "beta1_schedule": "MMCV one-cycle momentum",
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "seed": 42,
        },
        "loss": "SigLoss(1.0, unbiased) + GradientLoss(0.5)",
        "evaluation": "Eigen crop + horizontal-flip average; per-image RMSE",
        "train_dataset": dataset_protocol(train_dataset),
        "eval_dataset": dataset_protocol(eval_dataset),
        "vision_tower": tower.protocol_summary(),
    }


def main() -> None:
    args = parse_args()
    limits = run_limits(args.smoke)
    seed_everything(args.seed)
    train_dataset, eval_dataset = build_datasets(args, limits)
    tower = FrozenVisionTower(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    head = LinearDepthHead(
        tower.output_channels,
        patch_size=tower.metadata.patch_size,
    ).to(tower.device_name, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model_name = args.model_name or safe_model_name(args.model_id)
    run_name = f"{DATASET}_{TASK}_{model_name}"
    checkpoint_path = args.out_dir / "checkpoints" / f"{run_name}.pt"
    best_path = args.out_dir / "checkpoints" / f"{run_name}.best.pt"
    result_path = args.out_dir / f"{run_name}.json"
    protocol = protocol_summary(tower, train_dataset, eval_dataset, limits)
    print(json.dumps(protocol, indent=2, sort_keys=True))

    eval_loader = evaluation_loader(
        eval_dataset,
        args.eval_batch_size,
        args.num_workers,
    )
    start_step = 0
    best_rmse = float("inf")
    history: list[dict[str, Any]] = []
    if (args.resume or args.evaluate_only) and checkpoint_path.is_file():
        start_step, best_rmse, history = restore_checkpoint(
            checkpoint_path,
            head,
            optimizer,
            protocol,
        )
        print(f"[resume] step={start_step} checkpoint={checkpoint_path}")

    if args.evaluate_only:
        if start_step == 0:
            raise FileNotFoundError(checkpoint_path)
        metrics = evaluate(tower, head, eval_loader, args)
        payload = {
            "model_name": model_name,
            "model_id": args.model_id,
            "step": start_step,
            "metrics": asdict(metrics),
            "paper_reference": paper_reference(args.model_id, DATASET, TASK),
            "protocol": protocol,
            "history": history,
        }
        write_json(result_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    batches = iter(
        train_loader(
            train_dataset,
            limits.batch_size,
            args.seed,
            start_step,
            limits.steps,
            args.num_workers,
        )
    )
    running_loss = 0.0
    running_count = 0
    interval_started = time.monotonic()
    final_metrics: GeometryMetrics | None = None
    for step in range(start_step, limits.steps):
        batch = next(batches)
        learning_rate = learning_rate_at_step(
            step,
            limits.steps,
            LEARNING_RATE,
            WARMUP_STEPS,
            MIN_LR_FACTOR,
            WARMUP_START_FACTOR,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            group["betas"] = (
                one_cycle_beta1_at_step(step, limits.steps),
                group["betas"][1],
            )

        target = batch["depth"].to(tower.device_name, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(tower, head, batch["image"], target.shape[-2:], args)
        loss = loss_at_step(prediction, target, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        running_loss += float(loss.detach())
        running_count += 1
        completed_step = step + 1
        if completed_step % limits.log_interval == 0 or completed_step == limits.steps:
            elapsed = time.monotonic() - interval_started
            print(
                f"[train] step={completed_step}/{limits.steps} "
                f"loss={running_loss / running_count:.6f} "
                f"lr={learning_rate:.8g} time={elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0
            interval_started = time.monotonic()

        if completed_step % limits.eval_interval == 0 or completed_step == limits.steps:
            final_metrics = evaluate(tower, head, eval_loader, args)
            record = {"step": completed_step, **asdict(final_metrics)}
            history.append(record)
            print(f"[eval] {json.dumps(record, sort_keys=True)}", flush=True)
            if final_metrics.rmse < best_rmse:
                best_rmse = final_metrics.rmse
                atomic_torch_save(
                    checkpoint_payload(
                        head,
                        optimizer,
                        completed_step,
                        best_rmse,
                        history,
                        protocol,
                    ),
                    best_path,
                )
            head.train()

        if completed_step % limits.save_interval == 0 or completed_step == limits.steps:
            atomic_torch_save(
                checkpoint_payload(
                    head,
                    optimizer,
                    completed_step,
                    best_rmse,
                    history,
                    protocol,
                ),
                checkpoint_path,
            )

    if final_metrics is None:
        final_metrics = evaluate(tower, head, eval_loader, args)
    payload = {
        "model_name": model_name,
        "model_id": args.model_id,
        "step": limits.steps,
        "metrics": asdict(final_metrics),
        "best_rmse": best_rmse,
        "paper_reference": paper_reference(args.model_id, DATASET, TASK),
        "protocol": protocol,
        "history": history,
    }
    write_json(result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
