#!/usr/bin/env python3
"""Shared helpers for CLIP/SigLIP downstream representation probes."""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def parse_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None or name == "auto":
        return None
    normalized = name.lower()
    if normalized in {"none", "fp32", "float32"}:
        return None
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {name}")


def infer_model_family(model_id: str, config_model_type: str | None = None) -> str:
    text = f"{model_id} {config_model_type or ''}".lower()
    if "siglip2" in text:
        return "siglip2"
    if "siglip" in text:
        return "siglip"
    if "clip" in text:
        return "clip"
    return config_model_type or "unknown"


def from_pretrained_with_local_fallback(loader, model_id: str, *, local_files_only: bool, **kwargs):
    try:
        return loader.from_pretrained(model_id, local_files_only=local_files_only, **kwargs)
    except Exception as exc:
        if local_files_only:
            raise
        print(
            f"from_pretrained failed for {model_id} ({exc.__class__.__name__}); retrying local cache",
            file=sys.stderr,
        )
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)


def load_model_bundle(
    model_id: str,
    processor_id: str | None = None,
    tokenizer_id: str | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
    local_files_only: bool = False,
):
    device = resolve_device(device)
    processor_id = processor_id or model_id
    tokenizer_id = tokenizer_id or processor_id
    dtype = parse_torch_dtype(torch_dtype)
    processor = from_pretrained_with_local_fallback(
        AutoImageProcessor, processor_id, local_files_only=local_files_only
    )
    tokenizer = from_pretrained_with_local_fallback(
        AutoTokenizer, tokenizer_id, local_files_only=local_files_only
    )
    model = from_pretrained_with_local_fallback(
        AutoModel, model_id, local_files_only=local_files_only, torch_dtype=dtype
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    family = infer_model_family(model_id, getattr(model.config, "model_type", None))
    return {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        "model_id": model_id,
        "processor_id": processor_id,
        "tokenizer_id": tokenizer_id,
        "device": device,
        "family": family,
    }


def vision_config(model) -> Any:
    cfg = getattr(model.config, "vision_config", None)
    if cfg is None:
        raise ValueError("Model config does not expose vision_config")
    return cfg


def patch_grid(model) -> tuple[int, int, int, int]:
    cfg = vision_config(model)
    image_size = int(cfg.image_size)
    patch_size = int(cfg.patch_size)
    grid = image_size // patch_size
    hidden = int(cfg.hidden_size)
    if grid <= 0:
        raise ValueError(f"Invalid grid from image_size={image_size}, patch_size={patch_size}")
    return image_size, patch_size, grid, hidden


def _size_mapping(size: Any) -> dict[str, Any] | None:
    if is_dataclass(size):
        return asdict(size)
    if isinstance(size, Mapping):
        return dict(size)
    return None


def _size_hw(size: Any) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return size, size
    size_mapping = _size_mapping(size)
    if size_mapping is not None:
        if size_mapping.get("height") is not None and size_mapping.get("width") is not None:
            return int(size_mapping["height"]), int(size_mapping["width"])
        if size_mapping.get("shortest_edge") is not None:
            edge = int(size_mapping["shortest_edge"])
            return edge, edge
    return None


def resize_like_processor(image: Image.Image, processor, resample) -> Image.Image:
    """Apply the processor's resize and center-crop geometry to a label image.

    CLIP processors usually resize the shortest edge and center-crop. SigLIP
    processors resize directly to a fixed square without center-cropping. The
    distinction matters for dense labels and was a source of bad SigLIP probes
    in the older rebuttal scripts.
    """

    out = image
    size = getattr(processor, "size", None)
    if getattr(processor, "do_resize", False) and size is not None:
        width, height = out.size
        size_mapping = _size_mapping(size)
        shortest_edge = None if size_mapping is None else size_mapping.get("shortest_edge")
        if shortest_edge is not None:
            shortest = int(shortest_edge)
            scale = shortest / min(width, height)
            new_w = int(round(width * scale))
            new_h = int(round(height * scale))
            out = out.resize((new_w, new_h), resample)
        else:
            hw = _size_hw(size)
            if hw is None:
                raise ValueError(f"Unsupported processor size: {size}")
            target_h, target_w = hw
            out = out.resize((target_w, target_h), resample)

    if getattr(processor, "do_center_crop", False):
        crop_hw = _size_hw(getattr(processor, "crop_size", None)) or _size_hw(size)
        if crop_hw is None:
            raise ValueError("Processor requests center crop but has no crop_size")
        crop_h, crop_w = crop_hw
        width, height = out.size
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        out = out.crop((left, top, left + crop_w, top + crop_h))
    return out


def resize_to_patch_grid(image: Image.Image, processor, grid: int, resample) -> Image.Image:
    """Align dense targets with valid, non-overlapping patch-embedding coverage.

    SigLIP 384/14 yields a 27x27 grid and leaves six trailing pixels
    unencoded, so those pixels must not be folded into the target grid.
    """

    target = resize_like_processor(image, processor, resample)
    patch_h = target.height // grid
    patch_w = target.width // grid
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Target {target.size} is smaller than patch grid {grid}x{grid}")
    covered_h = patch_h * grid
    covered_w = patch_w * grid
    if (covered_w, covered_h) != target.size:
        target = target.crop((0, 0, covered_w, covered_h))
    return target.resize((grid, grid), resample)


def transform_mask_to_grid(mask: Image.Image, processor, grid: int) -> torch.Tensor:
    label = resize_to_patch_grid(mask, processor, grid, Image.Resampling.NEAREST)
    return torch.from_numpy(np.array(label, dtype=np.int64))


def transform_scalar_to_grid(image: Image.Image, processor, grid: int, scale: float = 1.0) -> torch.Tensor:
    target = resize_to_patch_grid(image, processor, grid, Image.Resampling.BILINEAR)
    arr = np.array(target, dtype=np.float32) / float(scale)
    return torch.from_numpy(arr).unsqueeze(0)


def transform_rgb_vector_to_grid(image: Image.Image, processor, grid: int) -> torch.Tensor:
    target = resize_to_patch_grid(image.convert("RGB"), processor, grid, Image.Resampling.BILINEAR)
    arr = np.array(target, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return torch.nn.functional.normalize(tensor, dim=0, eps=1e-6)


def processor_geometry_summary(processor) -> dict[str, Any]:
    size = getattr(processor, "size", None)
    crop_size = getattr(processor, "crop_size", None)
    return {
        "processor_class": processor.__class__.__name__,
        "do_resize": getattr(processor, "do_resize", None),
        "size": _size_mapping(size) or size,
        "do_center_crop": getattr(processor, "do_center_crop", None),
        "crop_size": _size_mapping(crop_size) or crop_size,
        "resample": getattr(processor, "resample", None),
    }


def patch_features(model, processor, images: list[Image.Image], device: str) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        out = model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=False)
        tokens = out.last_hidden_state
        token_count = int(tokens.shape[1])
        side = int(math.sqrt(token_count))
        if side * side == token_count:
            patch = tokens
        else:
            patch = tokens[:, 1:, :]
            token_count = int(patch.shape[1])
            side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Patch tokens are not square: token_count={token_count}")
        post_layernorm = getattr(model.vision_model, "post_layernorm", None)
        if post_layernorm is not None:
            patch = post_layernorm(patch)
    return patch.float()


def assert_patch_grid(model, processor, images: list[Image.Image], device: str) -> None:
    _, _, grid, _ = patch_grid(model)
    feats = patch_features(model, processor, images, device)
    expected = grid * grid
    actual = int(feats.shape[1])
    if actual != expected:
        raise ValueError(f"Patch grid mismatch: expected {expected} tokens ({grid}x{grid}), got {actual}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_model_name(model_name: str) -> str:
    keep = []
    for char in model_name:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")
