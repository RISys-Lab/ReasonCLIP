#!/usr/bin/env python3
"""Frozen-encoder LocCa/SigLIP2 RefCOCO grounding probe.

This trains a random six-layer Base autoregressive decoder on image-expression
pairs from the standard mixed RefCOCO, RefCOCO+, and RefCOCOg train splits
while keeping the image encoder frozen. This full mix matches the reported
paper table; an image-disjoint clean mix is available as a leakage-free
diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from timm.optim import AdafactorBigVision
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import safe_model_name, seed_everything, write_json  # noqa: E402
from grounding import (  # noqa: E402
    C4Tokenizer,
    LocCaDecoder,
    box_iou_xyxy,
    decoder_loss,
    dequantize_box_lbrt,
    parse_box_string,
    quantize_box_xywh,
    shift_right,
    xywh_to_xyxy,
)
from grounding_data import (  # noqa: E402
    IMAGE_TRANSFORM_NAME,
    RefCOCOLocCaDataset,
    collate_grounding,
    prompt_tokens,
    training_tokens,
)
from grounding_cache import (  # noqa: E402
    CachedGroundingDataset,
    CompositeGroundingFeatureCache,
    GroundingFeatureCache,
    build_feature_cache,
    collate_cached,
)
from probe_utils import FrozenVisionTower  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "downstream_data"
DEFAULT_RECORDS_ROOT = DEFAULT_DATA_ROOT / "RefCOCOLocCa"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "COCO" / "train2014"
DEFAULT_TOKENIZER = DEFAULT_DATA_ROOT / "C4Tokenizer" / "cc_en.32000.sentencepiece.model"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "downstream"

EVAL_SPLITS = (
    ("refcoco", "val"),
    ("refcoco", "testA"),
    ("refcoco", "testB"),
    ("refcocoplus", "val"),
    ("refcocoplus", "testA"),
    ("refcocoplus", "testB"),
    ("refcocog", "val"),
    ("refcocog", "test"),
)

PAPER_BASELINES = {
    ("openai/clip-vit-large-patch14", 224): {
        "refcoco_val": 65.21,
        "refcoco_testA": 71.28,
        "refcoco_testB": 58.17,
        "refcocoplus_val": 57.53,
        "refcocoplus_testA": 66.44,
        "refcocoplus_testB": 47.77,
        "refcocog_val": 59.32,
        "refcocog_test": 60.24,
    },
    ("google/siglip-so400m-patch14-384", 384): {
        "refcoco_val": 67.66,
        "refcoco_testA": 74.12,
        "refcoco_testB": 62.36,
        "refcocoplus_val": 60.74,
        "refcocoplus_testA": 69.73,
        "refcocoplus_testB": 52.12,
        "refcocog_val": 62.61,
        "refcocog_test": 63.24,
    },
}


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--processor-id")
    parser.add_argument("--model-name")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--effective-batch-size", type=int, default=512)
    parser.add_argument("--backbone-batch-size", type=int, default=8)
    parser.add_argument("--decoder-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--loss-scope",
        choices=["box_suffix", "full_aref"],
        default="full_aref",
        help=(
            "full_aref follows the LocCa generative target and the validated probe; "
            "box_suffix is a conditional-REC diagnostic"
        ),
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    parser.add_argument("--save-interval-steps", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train",
        type=int,
        help="limit training image-expression pairs",
    )
    parser.add_argument("--max-eval", type=int)
    parser.add_argument(
        "--training-mix",
        choices=["clean", "full"],
        default="full",
        help=(
            "full matches the reported paper table; clean removes every held-out "
            "image across datasets for a leakage-free diagnostic"
        ),
    )
    parser.add_argument("--feature-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--cache-flush-interval", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def infer_resolution(tower: FrozenVisionTower) -> int:
    for value in (
        getattr(tower.processor, "crop_size", None),
        getattr(tower.processor, "size", None),
    ):
        if isinstance(value, dict):
            if value.get("height") and value.get("width"):
                height, width = int(value["height"]), int(value["width"])
                if height != width:
                    raise ValueError(f"Grounding expects square processor geometry, got {value}")
                return height
            if value.get("shortest_edge"):
                return int(value["shortest_edge"])
        height = getattr(value, "height", None)
        width = getattr(value, "width", None)
        if height and width:
            if int(height) != int(width):
                raise ValueError(f"Grounding expects square processor geometry, got {value}")
            return int(height)
        shortest = getattr(value, "shortest_edge", None)
        if shortest:
            return int(shortest)
    raise RuntimeError("Could not infer image resolution from processor; pass --resolution")


def autocast_context(device: str, dtype_name: str):
    if not device.startswith("cuda"):
        return nullcontext()
    lowered = dtype_name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if lowered in {"fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def cosine_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return peak * step / warmup_steps
    denominator = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def grounding_training_steps(
    examples_per_epoch: int,
    effective_batch_size: int,
    epochs: int,
) -> tuple[int, int]:
    if examples_per_epoch <= 0 or effective_batch_size <= 0 or epochs <= 0:
        raise ValueError("Grounding training shape values must be positive")
    steps_per_epoch = examples_per_epoch // effective_batch_size
    if steps_per_epoch == 0:
        raise ValueError("Grounding training set has fewer than one effective batch")
    return steps_per_epoch, steps_per_epoch * epochs


def grounding_implementation_assumptions(
    effective_batch_size: int,
    warmup_ratio: float,
) -> str:
    return (
        f"effective batch {effective_batch_size} and warmup ratio {warmup_ratio:g}; "
        "neither value is disclosed for the LocCa/SigLIP 2 frozen REC probe"
    )


def paper_baseline_for_run(
    args: argparse.Namespace,
    model_id: str,
    resolution: int,
    global_step: int,
    expected_global_steps: int,
) -> dict[str, float] | None:
    if (
        global_step != expected_global_steps
        or args.training_mix != "full"
        or args.loss_scope != "full_aref"
        or args.epochs != 50
        or args.max_train is not None
        or args.max_eval is not None
    ):
        return None
    return PAPER_BASELINES.get((model_id, resolution))


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def adafactor_parameter_groups(
    decoder: LocCaDecoder,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Match Big Vision default kernel-only weight-decay filter."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    names = {"decay": [], "no_decay": []}
    embedding_parameters = {"token_embedding.weight", "position_embedding"}
    for name, parameter in decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        use_decay = parameter.ndim >= 2 and name not in embedding_parameters
        target_parameters = decay if use_decay else no_decay
        target_names = names["decay" if use_decay else "no_decay"]
        target_parameters.append(parameter)
        target_names.append(name)
    if not decay or not no_decay:
        raise RuntimeError("LocCa Adafactor parameter grouping produced an empty group")
    grouped_ids = [id(parameter) for parameter in decay + no_decay]
    expected_ids = [
        id(parameter) for parameter in decoder.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(expected_ids):
        raise RuntimeError("LocCa Adafactor parameter grouping is incomplete or duplicated")
    return (
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        names,
    )


def checkpoint_payload(
    decoder: LocCaDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    best_score: float,
) -> dict[str, Any]:
    return {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": protocol,
        "best_score": float(best_score),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def protocols_compatible_for_resume(
    saved: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    if saved == expected:
        return True

    def without_nonbehavioral_metadata(protocol: dict[str, Any]) -> dict[str, Any]:
        stripped = dict(protocol)
        stripped.pop("paper_disclosure", None)
        decoder_target = dict(stripped.get("decoder_target", {}))
        for key in (
            "conditional_box_suffix_default",
            "conditional_box_suffix_diagnostic",
            "full_aref_default",
            "full_aref_pretraining_diagnostic",
        ):
            decoder_target.pop(key, None)
        stripped["decoder_target"] = decoder_target
        record_files = dict(stripped.get("record_files", {}))
        record_files.pop("manifest", None)
        stripped["record_files"] = record_files
        return stripped

    return without_nonbehavioral_metadata(saved) == without_nonbehavioral_metadata(expected)


def restore_checkpoint(
    path: Path,
    decoder: LocCaDecoder,
    optimizer: torch.optim.Optimizer,
    load_optimizer: bool,
    expected_protocol: dict[str, Any],
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_protocol = payload.get("protocol")
    if not isinstance(saved_protocol, dict) or not protocols_compatible_for_resume(
        saved_protocol, expected_protocol
    ):
        raise RuntimeError(f"Grounding checkpoint protocol mismatch: {path}")
    if saved_protocol != expected_protocol:
        print(
            "[resume] accepting nonbehavioral metadata evolution; all actual "
            "grounding protocol fields and record fingerprints are unchanged",
            flush=True,
        )
    decoder.load_state_dict(payload["decoder"], strict=True)
    if load_optimizer:
        optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng")
    if rng and load_optimizer:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(payload.get("epoch", 0)), int(payload.get("global_step", 0)), float(payload.get("best_score", 0.0))


def create_loader(
    dataset: RefCOCOLocCaDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=drop_last,
        collate_fn=collate_grounding,
        generator=generator,
    )


def create_cached_loader(
    dataset: CachedGroundingDataset,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        drop_last=drop_last,
        collate_fn=collate_cached,
        generator=generator,
    )


def resume_epoch_position(
    global_step: int,
    optimizer_steps_per_epoch: int,
    accumulation_steps: int,
) -> tuple[int, int]:
    if global_step < 0 or optimizer_steps_per_epoch <= 0 or accumulation_steps <= 0:
        raise ValueError("Invalid grounding resume position")
    epoch, optimizer_step_in_epoch = divmod(global_step, optimizer_steps_per_epoch)
    return epoch, optimizer_step_in_epoch * accumulation_steps


@torch.no_grad()
def evaluate_split(
    tower: FrozenVisionTower,
    decoder: LocCaDecoder,
    tokenizer: C4Tokenizer,
    loader: DataLoader,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
    feature_cache: GroundingFeatureCache | None = None,
) -> dict[str, Any]:
    decoder.eval()
    total = 0
    correct = 0
    iou_sum = 0.0
    invalid = 0
    prompt_truncated = 0
    start = time.time()
    for batch in loader:
        if feature_cache is None:
            images = batch["images"].to(device, non_blocking=True)
        else:
            vision_tokens = feature_cache.get(batch["records"], device)
        with autocast_context(device, dtype_name):
            if feature_cache is None:
                vision_tokens = tower.sequence_features(tower.normalize(images))
            prompts, truncated = prompt_tokens(batch["records"], tokenizer)
            generated = decoder.generate(
                vision_tokens,
                prompts,
                pad_id=tokenizer.pad_id,
                eos_id=tokenizer.eos_id,
                max_new_tokens=max_new_tokens,
            )
        prompt_truncated += truncated
        for record, prompt, token_ids in zip(batch["records"], prompts, generated):
            suffix = tokenizer.decode(token_ids[len(prompt) :])
            prediction = parse_box_string(suffix)
            if prediction is None:
                iou = 0.0
                invalid += 1
            else:
                predicted_xyxy = dequantize_box_lbrt(
                    prediction,
                    record.width,
                    record.height,
                )
                iou = box_iou_xyxy(predicted_xyxy, xywh_to_xyxy(record.bbox_xywh))
            total += 1
            correct += int(iou >= 0.5)
            iou_sum += iou
    if total == 0:
        raise RuntimeError("Grounding evaluation produced no examples")
    return {
        "expressions": total,
        "acc_iou_0_5": 100.0 * correct / total,
        "mean_iou": 100.0 * iou_sum / total,
        "invalid_predictions": invalid,
        "invalid_rate": 100.0 * invalid / total,
        "prompt_truncated": prompt_truncated,
        "seconds": time.time() - start,
    }


def evaluation_datasets(
    args: argparse.Namespace,
    resolution: int,
    splits: Sequence[tuple[str, str]],
    feature_cache: GroundingFeatureCache | None = None,
) -> Iterator[tuple[str, RefCOCOLocCaDataset | CachedGroundingDataset]]:
    max_eval = args.max_eval
    for dataset, split in splits:
        name = f"{dataset}_{split}"
        records_path = args.records_root / f"{name}.jsonl"
        if feature_cache is None:
            split_dataset = RefCOCOLocCaDataset(
                records_path,
                args.image_root,
                resolution,
                max_samples=max_eval,
            )
        else:
            split_dataset = CachedGroundingDataset(records_path, max_samples=max_eval)
        yield name, split_dataset


def run_evaluation(
    args: argparse.Namespace,
    tower: FrozenVisionTower,
    decoder: LocCaDecoder,
    tokenizer: C4Tokenizer,
    resolution: int,
    splits: Sequence[tuple[str, str]],
    feature_cache: GroundingFeatureCache | None = None,
) -> dict[str, dict[str, Any]]:
    metrics = {}
    for name, dataset in evaluation_datasets(args, resolution, splits, feature_cache):
        if feature_cache is None:
            loader = create_loader(
                dataset,
                args.eval_batch_size,
                args.num_workers,
                shuffle=False,
                drop_last=False,
            )
        else:
            loader = create_cached_loader(
                dataset,
                args.eval_batch_size,
                args.num_workers,
                shuffle=False,
                drop_last=False,
            )
        metrics[name] = evaluate_split(
            tower,
            decoder,
            tokenizer,
            loader,
            tower.device_name,
            args.torch_dtype,
            args.max_new_tokens,
            feature_cache=feature_cache,
        )
        print(f"[eval] {name} Acc@0.5={metrics[name]['acc_iou_0_5']:.2f} mIoU={metrics[name]['mean_iou']:.2f}", flush=True)
    return metrics


def count_coordinate_suffix_truncations(
    records: Sequence[Any],
    tokenizer: C4Tokenizer,
    batch_size: int = 4096,
) -> int:
    count = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        boxes = [
            quantize_box_xywh(record.bbox_xywh, record.width, record.height)
            for record in batch
        ]
        _, _, truncated = training_tokens(batch, boxes, tokenizer)
        count += truncated
    return count


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.effective_batch_size = min(args.effective_batch_size, 8)
        args.backbone_batch_size = min(args.backbone_batch_size, 2)
        args.decoder_batch_size = min(args.decoder_batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.feature_cache = False
        args.max_train = args.max_train or 8
        args.max_eval = args.max_eval or 4
        args.eval_interval_epochs = 1
        args.save_interval_steps = 1
    training_batch_size = (
        args.decoder_batch_size if args.feature_cache else args.backbone_batch_size
    )
    if args.effective_batch_size % training_batch_size != 0:
        raise ValueError(
            "--effective-batch-size must be divisible by the active training micro-batch size"
        )
    seed_everything(args.seed)

    tower = FrozenVisionTower(
        args.model_id,
        processor_id=args.processor_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    resolution = args.resolution or infer_resolution(tower)
    if resolution < tower.metadata.patch_size:
        raise ValueError(
            f"Resolution {resolution} is smaller than patch size {tower.metadata.patch_size}"
        )
    tokenizer = C4Tokenizer(args.tokenizer)
    decoder = LocCaDecoder(
        vision_size=tower.metadata.hidden_size,
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        dropout=args.decoder_dropout,
    ).to(tower.device_name)
    optimizer_groups, optimizer_group_names = adafactor_parameter_groups(
        decoder,
        args.weight_decay,
    )
    optimizer = AdafactorBigVision(
        optimizer_groups,
        lr=args.learning_rate,
        min_dim_size_to_factor=32,
        decay_rate=0.8,
        beta2_cap=0.999,
        momentum=0.9,
        momentum_dtype=torch.bfloat16,
        eps=1e-30,
        weight_decay=0.0,
        clipping_threshold=None,
        foreach=False,
    )

    model_name = args.model_name or safe_model_name(args.model_id)
    run_name = (
        model_name
        if args.training_mix == "clean"
        else f"{model_name}_{args.training_mix}"
    )
    checkpoint_path = args.out_dir / "checkpoints" / "grounding" / f"{run_name}.pt"
    result_path = args.out_dir / "grounding" / f"{run_name}.json"
    training_record_name = f"train_{args.training_mix}"
    training_records_path = args.records_root / f"{training_record_name}.jsonl"
    if args.feature_cache:
        train_dataset = CachedGroundingDataset(
            training_records_path,
            max_samples=args.max_train,
            seed=args.seed,
        )
    else:
        train_dataset = RefCOCOLocCaDataset(
            training_records_path,
            args.image_root,
            resolution,
            max_samples=args.max_train,
            seed=args.seed,
        )
    training_sampling = train_dataset.sampling_summary()
    expected_steps_per_epoch, expected_global_steps = grounding_training_steps(
        len(train_dataset),
        args.effective_batch_size,
        args.epochs,
    )
    record_names = [training_record_name, "eval_all"] + [
        f"{dataset}_{split}" for dataset, split in EVAL_SPLITS
    ]
    if args.training_mix == "full":
        record_names.insert(1, "train_clean")
    record_fingerprints = {
        name: file_fingerprint(args.records_root / f"{name}.jsonl")
        for name in record_names
    }
    record_fingerprints["manifest"] = file_fingerprint(args.records_root / "manifest.json")
    protocol = {
        "name": "LocCa/SigLIP2 frozen-encoder RefCOCO REC",
        "vision": tower.protocol_summary(),
        "vision_tokens": (
            "unpooled normalized full sequence before pooling; OpenAI CLIP CLS included"
            if tower.metadata.family == "clip"
            else "unpooled normalized patch sequence before pooling; CLS/MAP excluded"
        ),
        "vision_sequence_padding": "none; convolution uses only complete patches",
        "resolution": resolution,
        "patch_sequence_length": (resolution // tower.metadata.patch_size) ** 2,
        (
            "clean_training_mix"
            if args.training_mix == "clean"
            else "full_training_mix"
        ): (
            "leakage-free diagnostic: RefCOCO-UNC + RefCOCO+-UNC + "
            "RefCOCOg-UMD with all held-out images removed"
            if args.training_mix == "clean"
            else (
                "paper table protocol: standard RefCOCO-UNC + RefCOCO+-UNC + "
                "RefCOCOg-UMD train splits; cross-dataset train/evaluation image "
                "overlap retained"
            )
        ),
        "training_sampling": training_sampling,
        "record_files": record_fingerprints,
        "box_format": "[left, bottom, right, top], integer coordinates in [0,500]",
        "tokenizer": {
            "name": "C4 English SentencePiece 32k",
            "path": str(tokenizer.model_path),
            "sha256": tokenizer.model_sha256,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "eos_id": tokenizer.eos_id,
            "lowercase": True,
            "sticky_eos": True,
            "max_length": tokenizer.max_length,
        },
        "decoder_target": {
            "sequence": '"ARef: {expression} : [left, bottom, right, top]"',
            "teacher_forcing": True,
            "loss_scope": args.loss_scope,
            "full_aref_default": (
                "loss covers expression, box, and EOS after the ARef task prefix"
            ),
            "conditional_box_suffix_diagnostic": (
                "expression is decoder input; loss covers box suffix and EOS only"
            ),
        },
        "decoder": {
            "layers": 6,
            "hidden_size": 768,
            "heads": 12,
            "mlp_size": 3072,
            "dropout": args.decoder_dropout,
            "input_output_embeddings_tied": False,
        },
        "optimizer": {
            "name": "ScalingViT Adafactor (timm AdafactorBigVision)",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "weight_decay_filter": "matrix kernels except token/position embeddings (.*/kernel$)",
            "weight_decay_parameter_names": optimizer_group_names["decay"],
            "no_weight_decay_parameter_names": optimizer_group_names["no_decay"],
            "min_dim_size_to_factor": 32,
            "decay_rate": 0.8,
            "beta2_cap": 0.999,
            "momentum": 0.9,
            "momentum_dtype": "bfloat16",
            "epsilon": 1e-30,
            "factor_clipping_threshold": None,
            "gradient_clip_norm": args.grad_clip_norm,
            "schedule": "cosine with linear warmup",
            "schedule_step_index": (
                "zero-based completed-update count; first update uses schedule(0)"
            ),
            "warmup_ratio": args.warmup_ratio,
        },
        "epochs": args.epochs,
        "seed": args.seed,
        "training_expression_limit": args.max_train,
        "evaluation_subset_limit": args.max_eval,
        "effective_batch_size": args.effective_batch_size,
        "training_micro_batch_size": training_batch_size,
        "frozen_feature_cache": {
            "enabled": args.feature_cache,
            "storage": "exact BF16 bit patterns in a resumable uint16 memmap",
            "coverage": (
                "clean training images and all held-out evaluation images"
                if args.training_mix == "clean"
                else "full-mix images via clean/evaluation composite caches"
            ),
            "backbone_batch_size": args.backbone_batch_size,
        },
        "image_transform": IMAGE_TRANSFORM_NAME,
        "generation": {
            "decoding": "greedy autoregressive",
            "max_new_tokens": args.max_new_tokens,
            "evaluation_splits": [
                f"{dataset}_{split}" for dataset, split in EVAL_SPLITS
            ],
        },
        "label_smoothing": args.label_smoothing,
        "resume_semantics": (
            "epoch-seeded image-expression-pair shuffle plus exact optimizer-step "
            "microbatch offset"
        ),
        "paper_disclosure": {
            "explicit": (
                "mixed RefCOCO variants, 6 decoder layers, lr=3e-4, resolution "
                "follows sequence length"
            ),
            "inherited_from_CapPa_LiT": (
                "concat image-question pairs, 50 pair-level epochs, AdaFactor, "
                "wd=1e-4, dropout=0.1, and label smoothing=0.1"
            ),
            "implementation_assumptions": grounding_implementation_assumptions(
                args.effective_batch_size,
                args.warmup_ratio,
            ),
            "table_interpretation": (
                "standard per-dataset train splits are concatenated without global "
                "held-out-image filtering; the reported table is not marked clean"
            ),
            "target_interpretation": (
                "LocCa explicitly applies full loss to its pretraining ARef target; "
                "downstream probe masking is not separately disclosed, while full "
                "ARef outperformed box-suffix supervision at epoch 10"
            ),
            "augmentation_limit": (
                "LocCa does not disclose whether or how the inherited LiT-Decoder "
                "Inception crop transforms REC boxes; this implementation uses "
                "deterministic resize-only preprocessing"
            ),
            "requested_mix": args.training_mix,
        },
    }
    print(json.dumps({"model": model_name, "protocol": protocol}, indent=2, sort_keys=True), flush=True)

    start_epoch = 0
    global_step = 0
    best_score = 0.0
    if checkpoint_path.is_file() and (args.resume or args.evaluate_only):
        start_epoch, global_step, best_score = restore_checkpoint(
            checkpoint_path,
            decoder,
            optimizer,
            load_optimizer=not args.evaluate_only,
            expected_protocol=protocol,
        )
        print(f"Resumed {checkpoint_path} at epoch={start_epoch} step={global_step}", flush=True)
    elif args.evaluate_only:
        raise FileNotFoundError(checkpoint_path)
    if global_step > expected_global_steps:
        raise RuntimeError(
            f"Checkpoint step {global_step} exceeds configured total {expected_global_steps}"
        )

    truncated_boxes_total = 0
    feature_cache: (
        GroundingFeatureCache | CompositeGroundingFeatureCache | None
    ) = None
    eval_feature_cache: GroundingFeatureCache | None = None
    if args.feature_cache:
        cache_base_name = f"{model_name}_r{resolution}_tfresize_v1"
        train_cache_name = cache_base_name
        if args.training_mix == "clean" and args.max_train is not None:
            train_cache_name += f"_n{args.max_train}"
        cache_root = (
            args.feature_cache_dir
            if args.feature_cache_dir is not None
            else args.out_dir / "feature_cache" / "grounding"
        )
        if not args.evaluate_only:
            cache_training_record_name = (
                training_record_name
                if args.training_mix == "clean"
                else "train_clean"
            )
            feature_cache = build_feature_cache(
                cache_root / train_cache_name,
                tower,
                args.records_root / f"{cache_training_record_name}.jsonl",
                args.image_root,
                resolution,
                args.backbone_batch_size,
                args.num_workers,
                lambda: autocast_context(tower.device_name, args.torch_dtype),
                max_samples=(
                    args.max_train if args.training_mix == "clean" else None
                ),
                flush_interval=args.cache_flush_interval,
            )
        eval_feature_cache = build_feature_cache(
            cache_root / f"{cache_base_name}_eval",
            tower,
            args.records_root / "eval_all.jsonl",
            args.image_root,
            resolution,
            args.backbone_batch_size,
            args.num_workers,
            lambda: autocast_context(tower.device_name, args.torch_dtype),
            flush_interval=args.cache_flush_interval,
        )
        if args.training_mix == "full" and feature_cache is not None:
            feature_cache = CompositeGroundingFeatureCache(
                [feature_cache, eval_feature_cache]
            )
        tower.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.evaluate_only:
        if isinstance(feature_cache, CompositeGroundingFeatureCache):
            feature_cache.validate_coverage(train_dataset.records)
        truncated_boxes_total = count_coordinate_suffix_truncations(
            train_dataset.records,
            tokenizer,
        )
        accumulation_steps = args.effective_batch_size // training_batch_size
        microbatches_per_epoch = len(train_dataset) // training_batch_size
        usable_microbatches = (
            microbatches_per_epoch // accumulation_steps * accumulation_steps
        )
        optimizer_steps_per_epoch = usable_microbatches // accumulation_steps
        if optimizer_steps_per_epoch != expected_steps_per_epoch:
            raise RuntimeError(
                "Grounding microbatch accounting disagrees with the effective-batch total: "
                f"{optimizer_steps_per_epoch} != {expected_steps_per_epoch}"
            )
        total_steps = expected_global_steps
        derived_epoch, resumed_micro_step = resume_epoch_position(
            global_step,
            optimizer_steps_per_epoch,
            accumulation_steps,
        )
        if start_epoch != derived_epoch:
            raise RuntimeError(
                f"Checkpoint epoch/step mismatch: stored epoch={start_epoch}, "
                f"step={global_step} implies epoch={derived_epoch}"
            )
        start_epoch = derived_epoch
        warmup_steps = round(total_steps * args.warmup_ratio)
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, args.epochs):
            train_dataset.set_epoch(epoch)
            epoch_generator = torch.Generator().manual_seed(args.seed + epoch)
            if feature_cache is not None:
                train_loader = create_cached_loader(
                    train_dataset,
                    args.decoder_batch_size,
                    args.num_workers,
                    generator=epoch_generator,
                )
            else:
                train_loader = create_loader(
                    train_dataset,
                    args.backbone_batch_size,
                    args.num_workers,
                    shuffle=True,
                    drop_last=True,
                    generator=epoch_generator,
                )
            first_micro_step = resumed_micro_step if epoch == start_epoch else 0
            epoch_start = time.time()
            running_loss_sum = 0.0
            running_example_count = 0
            gradient_example_count = 0
            optimizer_steps = 0
            for micro_step, batch in enumerate(train_loader):
                if micro_step < first_micro_step:
                    continue
                if micro_step >= usable_microbatches or global_step >= total_steps:
                    break
                labels, loss_mask, _ = training_tokens(
                    batch["records"],
                    batch["boxes_lbrt"],
                    tokenizer,
                    loss_scope=args.loss_scope,
                )
                active_examples = labels.shape[0]
                labels = labels.to(tower.device_name, non_blocking=True)
                loss_mask = loss_mask.to(tower.device_name, non_blocking=True)
                if feature_cache is not None:
                    vision_tokens = feature_cache.get(batch["records"], tower.device_name)
                else:
                    images = batch["images"].to(tower.device_name, non_blocking=True)
                    with torch.no_grad(), autocast_context(tower.device_name, args.torch_dtype):
                        vision_tokens = tower.sequence_features(tower.normalize(images))
                input_ids = shift_right(labels, tokenizer.pad_id)
                with autocast_context(tower.device_name, args.torch_dtype):
                    logits = decoder(vision_tokens, input_ids, pad_id=tokenizer.pad_id)
                    loss = decoder_loss(
                        logits,
                        labels,
                        loss_mask,
                        label_smoothing=args.label_smoothing,
                        reduction="sum",
                    )
                loss.backward()
                gradient_example_count += active_examples
                running_example_count += active_examples
                running_loss_sum += loss.detach().float()
                if (micro_step + 1) % accumulation_steps != 0:
                    continue
                learning_rate = cosine_learning_rate(
                    global_step,
                    total_steps,
                    warmup_steps,
                    args.learning_rate,
                )
                next_step = global_step + 1
                set_learning_rate(optimizer, learning_rate)
                if gradient_example_count <= 0:
                    raise RuntimeError("Effective grounding batch contains no examples")
                for parameter in decoder.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(gradient_example_count)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    args.grad_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gradient_example_count = 0
                global_step = next_step
                optimizer_steps += 1
                if global_step % args.log_interval == 0:
                    average_loss = float(running_loss_sum) / max(running_example_count, 1)
                    print(
                        f"[train] epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                        f"loss={average_loss:.6f} lr={learning_rate:.8f} grad_norm={float(grad_norm):.4f}",
                        flush=True,
                    )
                    running_loss_sum = 0.0
                    running_example_count = 0
                if global_step % args.save_interval_steps == 0:
                    save_checkpoint(
                        checkpoint_path,
                        checkpoint_payload(
                            decoder,
                            optimizer,
                            epoch,
                            global_step,
                            args,
                            protocol,
                            best_score,
                        ),
                    )
            print(
                f"[epoch] {epoch + 1}/{args.epochs} optimizer_steps={optimizer_steps} "
                f"seconds={time.time() - epoch_start:.1f}",
                flush=True,
            )
            completed_epoch = epoch + 1
            save_checkpoint(
                checkpoint_path,
                checkpoint_payload(
                    decoder,
                    optimizer,
                    completed_epoch,
                    global_step,
                    args,
                    protocol,
                    best_score,
                ),
            )
            if global_step >= total_steps:
                break
            if completed_epoch % args.eval_interval_epochs == 0 and completed_epoch < args.epochs:
                validation_splits = (("refcoco", "val"), ("refcocoplus", "val"), ("refcocog", "val"))
                validation = run_evaluation(
                    args,
                    tower,
                    decoder,
                    tokenizer,
                    resolution,
                    validation_splits,
                    feature_cache=eval_feature_cache,
                )
                score = float(np.mean([item["acc_iou_0_5"] for item in validation.values()]))
                best_score = max(best_score, score)
                decoder.train()
        if truncated_boxes_total:
            print(
                f"[tokenizer] sticky-EOS truncation can affect the coordinate suffix in "
                f"{truncated_boxes_total} candidate sentences",
                flush=True,
            )

    if eval_feature_cache is None:
        tower.to(tower.device_name)
    metrics = run_evaluation(
        args,
        tower,
        decoder,
        tokenizer,
        resolution,
        EVAL_SPLITS,
        feature_cache=eval_feature_cache,
    )
    training_complete = global_step == expected_global_steps
    paper = paper_baseline_for_run(
        args,
        tower.metadata.model_id,
        resolution,
        global_step,
        expected_global_steps,
    )
    comparison = None
    if paper is not None:
        comparison = {
            name: {
                "paper_acc_iou_0_5": paper[name],
                "reproduced_acc_iou_0_5": metrics[name]["acc_iou_0_5"],
                "delta": metrics[name]["acc_iou_0_5"] - paper[name],
            }
            for name in paper
        }
    result = {
        "model_name": model_name,
        "run_name": run_name,
        "training_mix": args.training_mix,
        "model_id": args.model_id,
        "processor_id": tower.processor_id,
        "protocol": protocol,
        "checkpoint": str(checkpoint_path.resolve()),
        "global_step": global_step,
        "expected_global_steps": expected_global_steps,
        "completed_epoch_equivalent": global_step / expected_steps_per_epoch,
        "training_complete": training_complete,
        "training_coordinate_suffix_truncated_candidates": truncated_boxes_total,
        "metrics": metrics,
        "paper_baseline_comparison": comparison,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {result_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
