#!/usr/bin/env python3
"""Shared model utilities for paper-protocol frozen-backbone probes."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from downstream_utils import (  # noqa: E402
    from_pretrained_with_local_fallback,
    parse_torch_dtype,
    resolve_device,
)


@dataclass(frozen=True)
class VisionMetadata:
    model_id: str
    processor_id: str
    family: str
    hidden_size: int
    patch_size: int
    num_hidden_layers: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    model_revision: str | None = None
    processor_revision: str | None = None


def _snapshot_revision_from_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    parts = Path(path).parts
    try:
        snapshot_index = parts.index("snapshots")
    except ValueError:
        return None
    revision_index = snapshot_index + 1
    if revision_index >= len(parts):
        return None
    return parts[revision_index]


def _cached_snapshot_revision(repo_id: str, filename: str) -> str | None:
    direct_revision = _snapshot_revision_from_path(repo_id)
    if direct_revision is not None:
        return direct_revision
    if Path(repo_id).exists():
        return None
    try:
        cached_path = try_to_load_from_cache(repo_id, filename)
    except (OSError, ValueError):
        return None
    if not isinstance(cached_path, str):
        return None
    return _snapshot_revision_from_path(cached_path)


def _fold_seed(seed: int, epoch: int, position: int) -> int:
    mask = (1 << 63) - 1
    value = int(seed) & mask
    value ^= ((int(epoch) + 1) * 0x1E35A7BD) & mask
    value ^= ((int(position) + 1) * 0x5DEECE66D) & mask
    return value & mask


class DeterministicBatchSampler:
    """Finite shuffled batch stream addressable by absolute batch index."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_batch: int,
        num_batches: int,
        with_sample_seed: bool = False,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("Dataset and batch sizes must be positive")
        if start_batch < 0 or num_batches < 0:
            raise ValueError("Batch offsets must be non-negative")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_batch = int(start_batch)
        self.num_batches = int(num_batches)
        self.with_sample_seed = bool(with_sample_seed)
        self.batches_per_epoch = self.dataset_size // self.batch_size
        if self.batches_per_epoch == 0:
            raise ValueError("Dataset has fewer samples than one full batch")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int | tuple[int, int]]]:
        active_epoch = -1
        permutation: list[int] = []
        for absolute_batch in range(
            self.start_batch,
            self.start_batch + self.num_batches,
        ):
            epoch, batch_in_epoch = divmod(absolute_batch, self.batches_per_epoch)
            if epoch != active_epoch:
                generator = torch.Generator().manual_seed(_fold_seed(self.seed, epoch, 0))
                permutation = torch.randperm(
                    self.dataset_size,
                    generator=generator,
                ).tolist()
                active_epoch = epoch
            begin = batch_in_epoch * self.batch_size
            batch = permutation[begin : begin + self.batch_size]
            if self.with_sample_seed:
                yield [
                    (index, _fold_seed(self.seed, epoch, begin + offset + 1))
                    for offset, index in enumerate(batch)
                ]
            else:
                yield batch


class DeterministicAugmentDataset(Dataset):
    """Run a dataset item with a sampler-provided Python/NumPy/Torch seed."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: tuple[int, int]) -> Any:
        index, seed = key
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        try:
            return self.dataset[int(index)]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


def _three_floats(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"Expected three {name} values, got {value}")
    return tuple(float(item) for item in value)


def _model_family(model: nn.Module) -> str:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "siglip" in model_type:
        return "siglip"
    if "clip" in model_type:
        return "clip"
    raise ValueError(f"Unsupported model type for official probes: {model_type}")


class FrozenVisionTower(nn.Module):
    """Frozen CLIP/SigLIP vision tower with arbitrary-resolution features.

    Dense protocols pad images symmetrically to a patch-size multiple, matching
    DINOv2 and Probe3D. SigLIP's MAP embedding is used as its global token.
    """

    def __init__(
        self,
        model_id: str,
        processor_id: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = "bf16",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.device_name = resolve_device(device)
        self.processor_id = processor_id or model_id
        dtype = parse_torch_dtype(torch_dtype)

        self.processor = from_pretrained_with_local_fallback(
            AutoImageProcessor,
            self.processor_id,
            local_files_only=local_files_only,
        )
        full_model = from_pretrained_with_local_fallback(
            AutoModel,
            model_id,
            local_files_only=local_files_only,
            dtype=dtype,
        )
        if not hasattr(full_model, "vision_model"):
            raise ValueError(f"{model_id} does not expose vision_model")

        family = _model_family(full_model)
        cfg = full_model.config.vision_config
        self.vision_model = full_model.vision_model.to(self.device_name).eval()
        for parameter in self.vision_model.parameters():
            parameter.requires_grad_(False)

        mean = _three_floats(getattr(self.processor, "image_mean"), "image_mean")
        std = _three_floats(getattr(self.processor, "image_std"), "image_std")
        self.metadata = VisionMetadata(
            model_id=model_id,
            processor_id=self.processor_id,
            family=family,
            hidden_size=int(cfg.hidden_size),
            patch_size=int(cfg.patch_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            image_mean=mean,
            image_std=std,
            model_revision=(
                getattr(full_model.config, "_commit_hash", None)
                or _cached_snapshot_revision(model_id, "config.json")
            ),
            processor_revision=(
                getattr(self.processor, "_commit_hash", None)
                or _cached_snapshot_revision(self.processor_id, "preprocessor_config.json")
            ),
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32, device=self.device_name).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32, device=self.device_name).view(1, 3, 1, 1),
            persistent=False,
        )

        # The full multimodal container is no longer needed; retaining only the
        # vision tower avoids placing the unused text encoder on GPU.
        del full_model

    @property
    def output_channels(self) -> int:
        return 2 * self.metadata.hidden_size

    def default_dpt_layer_indices(self) -> tuple[int, int, int, int]:
        count = self.metadata.num_hidden_layers
        return (
            count // 4 - 1,
            count // 2 - 1,
            count * 3 // 4 - 1,
            count - 1,
        )

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.image_mean) / self.image_std

    def center_pad(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        patch = self.metadata.patch_size
        height, width = images.shape[-2:]
        target_h = math.ceil(height / patch) * patch
        target_w = math.ceil(width / patch) * patch
        pad_h = target_h - height
        pad_w = target_w - width
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        if pad_h or pad_w:
            images = F.pad(images, (left, right, top, bottom), value=0.0)
        return images, (left, right, top, bottom)

    def _split_patches(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        expected = grid_h * grid_w
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:]
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"Expected {expected} patch tokens for {grid_h}x{grid_w}, "
                f"got {tokens.shape[1]}"
            )
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)

    def final_features(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Return final patch features concatenated with global MAP/CLS."""

        images, _ = self.center_pad(normalized_images)
        grid_h = images.shape[-2] // self.metadata.patch_size
        grid_w = images.shape[-1] // self.metadata.patch_size
        outputs = self.vision_model(
            pixel_values=images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        patch_tokens = outputs.last_hidden_state
        if self.metadata.family == "clip":
            # TIPS/DINOv2 dense probes request norm=True for all tokens.
            patch_tokens = self.vision_model.post_layernorm(patch_tokens)
        patches = self._split_patches(patch_tokens, grid_h, grid_w)
        global_token = outputs.pooler_output
        if global_token is None:
            raise RuntimeError("Vision model did not return a global pooled embedding")
        global_map = global_token[:, :, None, None].expand(-1, -1, grid_h, grid_w)
        return torch.cat((patches, global_map), dim=1)

    def sequence_features(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Return the complete unpooled sequence used by the LocCa decoder.

        OpenAI CLIP exposes a CLS token and applies its final LayerNorm only to
        that token in Transformers, so normalize the complete sequence here.
        CapPa explicitly retains this CLS token when probing OpenAI CLIP.
        SigLIP has no CLS token and already returns post-LayerNorm patch tokens.
        The SigLIP MAP output is intentionally excluded: LocCa cross-attends to
        the vision sequence before the MAP head.
        """

        images = normalized_images
        grid_h = images.shape[-2] // self.metadata.patch_size
        grid_w = images.shape[-1] // self.metadata.patch_size
        outputs = self.vision_model(
            pixel_values=images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state
        expected_patches = grid_h * grid_w
        if self.metadata.family == "clip":
            tokens = self.vision_model.post_layernorm(tokens)
            if tokens.shape[1] != expected_patches + 1:
                raise RuntimeError(
                    f"Expected CLS plus {expected_patches} grounding patches, "
                    f"got {tokens.shape[1]}"
                )
        elif tokens.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} grounding patch tokens for "
                f"{grid_h}x{grid_w} {self.metadata.family}, got {tokens.shape[1]}"
            )
        return tokens

    def intermediate_features(
        self,
        normalized_images: torch.Tensor,
        layer_indices: Sequence[int] | None = None,
        concatenate_global: bool = True,
        normalize_intermediate: bool = True,
    ) -> list[torch.Tensor]:
        """Return four uniformly spaced block outputs for DPT probes."""

        feature_pairs = self.dpt_features(
            normalized_images,
            layer_indices=layer_indices,
            normalize_intermediate=normalize_intermediate,
        )
        if concatenate_global:
            return [
                torch.cat(
                    (
                        patch_map,
                        global_token[:, :, None, None].expand_as(patch_map),
                    ),
                    dim=1,
                )
                for global_token, patch_map in feature_pairs
            ]
        return [patch_map for _, patch_map in feature_pairs]

    def dpt_features(
        self,
        normalized_images: torch.Tensor,
        layer_indices: Sequence[int] | None = None,
        normalize_intermediate: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return four global-token and patch-map pairs for the TIPS DPT."""

        images, _ = self.center_pad(normalized_images)
        grid_h = images.shape[-2] // self.metadata.patch_size
        grid_w = images.shape[-1] // self.metadata.patch_size
        if layer_indices is None:
            layer_indices = self.default_dpt_layer_indices()
        if len(layer_indices) != 4:
            raise ValueError(f"DPT requires four layer indices, got {layer_indices}")

        outputs = self.vision_model(
            pixel_values=images,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Vision model did not return hidden states")
        final_global_token = outputs.pooler_output
        if final_global_token is None:
            raise RuntimeError("Vision model did not return a global pooled embedding")
        post_layernorm = getattr(self.vision_model, "post_layernorm", None)
        features = []
        for layer_index in layer_indices:
            # hidden_states[0] is the embedding output.
            layer_tokens = outputs.hidden_states[layer_index + 1]
            if normalize_intermediate:
                if post_layernorm is None:
                    raise RuntimeError("Vision model has no post_layernorm for DPT features")
                layer_tokens = post_layernorm(layer_tokens)
            if self.metadata.family == "clip":
                layer_global_token = layer_tokens[:, 0]
            else:
                # SigLIP has no CLS token; append its final MAP output.
                layer_global_token = final_global_token
            patch_map = self._split_patches(layer_tokens, grid_h, grid_w)
            features.append((layer_global_token, patch_map))
        return features

    def protocol_summary(self) -> dict[str, Any]:
        return {
            **self.metadata.__dict__,
            "dense_center_pad_to_patch_multiple": True,
            "grounding_sequence_center_pad_to_patch_multiple": False,
            "grounding_sequence_patch_policy": (
                "complete convolutional patches plus the leading CLS token"
                if self.metadata.family == "clip"
                else "complete convolutional patches only"
            ),
            "default_dpt_layer_indices_zero_based": list(
                self.default_dpt_layer_indices()
            ),
            "default_dpt_layer_indices_one_based": [
                index + 1 for index in self.default_dpt_layer_indices()
            ],
            "global_feature": "map" if self.metadata.family == "siglip" else "cls",
            "final_feature_channels": self.output_channels,
        }
