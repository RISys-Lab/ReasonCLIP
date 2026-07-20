#!/usr/bin/env python3
"""NAVI frozen-backbone relative-depth evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
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
    extract_dpt_features,
    geometry_parser,
    gradient_loss,
    learning_rate_at_step,
    paper_reference,
    restore_checkpoint,
    run_limits,
    scenic_lecun_init,
    sig_loss,
    train_loader,
)
from geometry_data import NAVIProbeDataset, dataset_protocol  # noqa: E402
from probe_utils import FrozenVisionTower  # noqa: E402
from tips_dpt import DepthDecoder  # noqa: E402

DATASET = "navi"
TASK = "depth"
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 7_500
WARMUP_START_FACTOR = 0.01
MIN_LR_FACTOR = 0.01
SIG_LOSS_WEIGHT = 10.0
GRADIENT_LOSS_WEIGHT = 0.5


def parse_args() -> argparse.Namespace:
    return geometry_parser(__doc__).parse_args()


def build_datasets(args: argparse.Namespace, limits: RunLimits):
    root = args.data_root / "NAVI" / "probe3d_navi_v1" / "navi_v1"
    train_dataset = NAVIProbeDataset(
        root,
        "trainval",
        augment=True,
        relative_depth=True,
        max_samples=limits.max_train,
    )
    eval_dataset = NAVIProbeDataset(
        root,
        "test",
        augment=False,
        relative_depth=True,
        max_samples=limits.max_eval,
    )
    return train_dataset, eval_dataset


def predict(
    tower: FrozenVisionTower,
    head: DepthDecoder,
    images: torch.Tensor,
    target_size: tuple[int, int],
    args: argparse.Namespace,
) -> torch.Tensor:
    features = extract_dpt_features(
        tower,
        images,
        args.backbone_batch_size,
        args.torch_dtype,
    )
    with autocast_context(tower.device_name, args.torch_dtype):
        prediction = head(features, image_size=target_size)
    return prediction.float()


def depth_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.masked_fill(target > 1.0, 0)
    return sig_loss(
        prediction,
        target,
        SIG_LOSS_WEIGHT,
        warmup=False,
        unbiased_variance=False,
    ) + GRADIENT_LOSS_WEIGHT * gradient_loss(prediction, target)


def evaluate(
    tower: FrozenVisionTower,
    head: DepthDecoder,
    loader: DataLoader,
    args: argparse.Namespace,
) -> GeometryMetrics:
    head.eval()
    values: dict[str, list[torch.Tensor]] = {}
    with torch.inference_mode():
        for batch in loader:
            target = batch["depth"].to(tower.device_name, non_blocking=True)
            prediction = predict(
                tower,
                head,
                batch["image"],
                target.shape[-2:],
                args,
            )
            for name, metric in depth_metrics(prediction, target).items():
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
    train_dataset: NAVIProbeDataset,
    eval_dataset: NAVIProbeDataset,
    limits: RunLimits,
) -> dict[str, Any]:
    return {
        "benchmark": "navi_depth",
        "paper_protocol": "TIPS Appendix B + Probe3D NAVI",
        "frozen_backbone": True,
        "head": "TIPS 4-layer DPT -> 256 uniform bins in [0.001, 1.0]",
        "training": {
            "steps": limits.steps,
            "batch_size": limits.batch_size,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "warmup_start_factor": WARMUP_START_FACTOR,
            "min_lr_factor": MIN_LR_FACTOR,
            "seed": 42,
        },
        "loss": "Probe3D SigLoss(10.0, biased) + GradientLoss(0.5)",
        "evaluation": "per-image relative-depth RMSE; no flip or scale-shift",
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
    head = DepthDecoder(
        input_embed_dim=tower.metadata.hidden_size,
        min_depth=0.001,
        max_depth=1.0,
        output_activation=False,
    ).to(tower.device_name, dtype=torch.float32)
    head.apply(scenic_lecun_init)
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

        target = batch["depth"].to(tower.device_name, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(tower, head, batch["image"], target.shape[-2:], args)
        loss = depth_loss(prediction, target)
        loss.backward()
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
