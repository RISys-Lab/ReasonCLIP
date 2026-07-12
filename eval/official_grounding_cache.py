#!/usr/bin/env python3
"""Resumable BF16 feature cache for the frozen LocCa vision sequence."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from official_grounding import quantize_box_xywh
from official_grounding_data import (
    EpochGroundingRecordSampler,
    GroundingRecord,
    IMAGE_TRANSFORM_NAME,
    image_to_tensor,
    load_grounding_records,
)
from official_probe_utils import FrozenVisionTower


class UniqueGroundingImages(Dataset):
    def __init__(self, records: Sequence[GroundingRecord], image_root: Path, resolution: int) -> None:
        by_image: dict[int, str] = {}
        for record in records:
            previous = by_image.setdefault(record.image_id, record.file_name)
            if previous != record.file_name:
                raise RuntimeError(
                    f"Image ID {record.image_id} has conflicting files: {previous}, {record.file_name}"
                )
        self.items = sorted(by_image.items())
        self.image_root = image_root
        self.resolution = int(resolution)
        missing = [file_name for _, file_name in self.items if not (image_root / file_name).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} cache-source images under {image_root}; first: {preview}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        _, file_name = self.items[index]
        with Image.open(self.image_root / file_name) as source:
            image = image_to_tensor(source, self.resolution)
        return index, image


def image_list_hash(items: Sequence[tuple[int, str]]) -> str:
    digest = hashlib.sha256()
    for image_id, file_name in items:
        digest.update(f"{image_id}\t{file_name}\n".encode("utf-8"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _expected_bytes(shape: Sequence[int], dtype: np.dtype) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


class GroundingFeatureCache:
    """Read-only mapping from RefCOCO image IDs to preserved BF16 tokens."""

    def __init__(self, cache_dir: Path) -> None:
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not self.metadata.get("complete"):
            raise RuntimeError(f"Incomplete grounding feature cache: {cache_dir}")
        self.cache_dir = cache_dir
        self.shape = tuple(int(value) for value in self.metadata["shape"])
        self.items = [
            (int(item["image_id"]), str(item["file_name"]))
            for item in self.metadata["images"]
        ]
        self.index_by_image_id = {image_id: index for index, (image_id, _) in enumerate(self.items)}
        if len(self.index_by_image_id) != len(self.items):
            raise RuntimeError(f"Duplicate image IDs in feature cache {cache_dir}")
        feature_path = cache_dir / "features.bf16"
        expected = _expected_bytes(self.shape, np.dtype("uint16"))
        if feature_path.stat().st_size != expected:
            raise RuntimeError(
                f"Feature cache size mismatch: expected {expected}, got {feature_path.stat().st_size}"
            )
        self.features = np.memmap(feature_path, mode="r", dtype=np.uint16, shape=self.shape)

    def indices(self, records: Sequence[GroundingRecord]) -> np.ndarray:
        try:
            return np.asarray(
                [self.index_by_image_id[record.image_id] for record in records],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise KeyError(f"Image ID {exc.args[0]} is absent from grounding feature cache") from exc

    def get(self, records: Sequence[GroundingRecord], device: str) -> torch.Tensor:
        indices = self.indices(records)
        # Advanced indexing makes one contiguous copy, avoiding undefined writes
        # against a read-only mmap before viewing the uint16 BF16 payload.
        payload = np.array(self.features[indices], copy=True)
        tokens = torch.from_numpy(payload).view(torch.bfloat16)
        return tokens.to(device, non_blocking=True)


class CompositeGroundingFeatureCache:
    """Route records across disjoint feature caches without duplicating data."""

    def __init__(self, caches: Sequence[GroundingFeatureCache]) -> None:
        if not caches:
            raise ValueError("At least one grounding feature cache is required")
        self.caches = list(caches)
        reference = self.caches[0]
        self.shape = reference.shape[1:]
        compatibility_keys = (
            "version",
            "token_contract",
            "model_id",
            "model_revision",
            "processor_id",
            "processor_revision",
            "family",
            "resolution",
            "image_transform",
            "patch_size",
            "dtype",
        )
        self.location_by_image_id: dict[int, tuple[int, int]] = {}
        for cache_index, cache in enumerate(self.caches):
            if cache.shape[1:] != self.shape:
                raise RuntimeError("Composite grounding caches have incompatible feature shapes")
            for key in compatibility_keys:
                if cache.metadata.get(key) != reference.metadata.get(key):
                    raise RuntimeError(
                        f"Composite grounding caches disagree on {key}: "
                        f"{cache.metadata.get(key)!r} != {reference.metadata.get(key)!r}"
                    )
            for row, (image_id, _) in enumerate(cache.items):
                if image_id in self.location_by_image_id:
                    raise RuntimeError(
                        f"Image ID {image_id} occurs in multiple composite feature caches"
                    )
                self.location_by_image_id[image_id] = (cache_index, row)

    def validate_coverage(self, records: Sequence[GroundingRecord]) -> None:
        missing = sorted(
            {record.image_id for record in records}.difference(
                self.location_by_image_id
            )
        )
        if missing:
            raise RuntimeError(
                f"Composite grounding cache misses {len(missing)} image IDs; first={missing[:5]}"
            )

    def get(self, records: Sequence[GroundingRecord], device: str) -> torch.Tensor:
        payload = np.empty((len(records), *self.shape), dtype=np.uint16)
        positions_by_cache: list[list[int]] = [[] for _ in self.caches]
        rows_by_cache: list[list[int]] = [[] for _ in self.caches]
        for position, record in enumerate(records):
            try:
                cache_index, row = self.location_by_image_id[record.image_id]
            except KeyError as exc:
                raise KeyError(
                    f"Image ID {record.image_id} is absent from composite feature caches"
                ) from exc
            positions_by_cache[cache_index].append(position)
            rows_by_cache[cache_index].append(row)
        for cache, positions, rows in zip(
            self.caches,
            positions_by_cache,
            rows_by_cache,
        ):
            if positions:
                payload[np.asarray(positions)] = cache.features[np.asarray(rows)]
        tokens = torch.from_numpy(payload).view(torch.bfloat16)
        return tokens.to(device, non_blocking=True)


class CachedGroundingDataset(Dataset):
    def __init__(
        self,
        records_path: Path,
        max_samples: int | None = None,
        sample_one_per_image: bool = False,
        seed: int = 0,
    ) -> None:
        records = load_grounding_records(
            records_path,
            max_samples=None if sample_one_per_image else max_samples,
        )
        self.record_sampler = EpochGroundingRecordSampler(
            records,
            sample_one_per_image=sample_one_per_image,
            seed=seed,
            max_images=max_samples if sample_one_per_image else None,
        )
        self.records = self.record_sampler.records

    def __len__(self) -> int:
        return len(self.record_sampler)

    def set_epoch(self, epoch: int) -> None:
        self.record_sampler.set_epoch(epoch)

    def records_for_epoch(self, epoch: int | None = None) -> list[GroundingRecord]:
        return self.record_sampler.records_for_epoch(epoch)

    def sampling_summary(self) -> dict[str, Any]:
        return self.record_sampler.summary()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.record_sampler.record_at(index)
        return {
            "record": record,
            "box_lbrt": quantize_box_xywh(
                record.bbox_xywh,
                record.width,
                record.height,
            ),
        }


def collate_cached(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": [item["record"] for item in items],
        "boxes_lbrt": [item["box_lbrt"] for item in items],
    }


@torch.no_grad()
def build_feature_cache(
    cache_dir: Path,
    tower: FrozenVisionTower,
    records_path: Path,
    image_root: Path,
    resolution: int,
    batch_size: int,
    workers: int,
    autocast_factory: Callable[[], Any],
    max_samples: int | None = None,
    sample_one_per_image: bool = False,
    flush_interval: int = 50,
) -> GroundingFeatureCache:
    records = load_grounding_records(
        records_path,
        max_samples=None if sample_one_per_image else max_samples,
    )
    if sample_one_per_image:
        records = EpochGroundingRecordSampler(
            records,
            sample_one_per_image=True,
            max_images=max_samples,
        ).records
    image_dataset = UniqueGroundingImages(records, image_root, resolution)
    token_count = (resolution // tower.metadata.patch_size) ** 2 + int(
        tower.metadata.family == "clip"
    )
    shape = (len(image_dataset), token_count, tower.metadata.hidden_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    feature_path = cache_dir / "features.bf16"
    complete_path = cache_dir / "complete.uint8"
    clip_sequence = tower.metadata.family == "clip"
    expected_metadata = {
        # Preserve compatibility with existing SigLIP caches. CLIP v5 corrects
        # the earlier patch-only contract by retaining its pre-pooling CLS.
        "version": 5 if clip_sequence else 4,
        "token_contract": (
            "normalized complete-patch sequence plus leading CLS; no input padding; "
            "MAP excluded"
            if clip_sequence
            else "normalized complete-patch sequence only; no input padding; CLS/MAP excluded"
        ),
        "image_transform": IMAGE_TRANSFORM_NAME,
        "model_id": tower.metadata.model_id,
        "model_revision": tower.metadata.model_revision,
        "processor_id": tower.metadata.processor_id,
        "processor_revision": tower.metadata.processor_revision,
        "family": tower.metadata.family,
        "resolution": int(resolution),
        "patch_size": tower.metadata.patch_size,
        "dtype": "bfloat16-bit-pattern-as-uint16",
        "shape": list(shape),
        "image_list_sha256": image_list_hash(image_dataset.items),
        "images": [
            {"image_id": image_id, "file_name": file_name}
            for image_id, file_name in image_dataset.items
        ],
    }

    if metadata_path.is_file():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected_metadata.items():
            if current.get(key) != value:
                raise RuntimeError(
                    f"Feature cache metadata mismatch for {key}: {current.get(key)!r} != {value!r}"
                )
        if current.get("complete"):
            return GroundingFeatureCache(cache_dir)
        if feature_path.stat().st_size != _expected_bytes(shape, np.dtype("uint16")):
            raise RuntimeError(f"Partial feature file has the wrong size: {feature_path}")
        if complete_path.stat().st_size != len(image_dataset):
            raise RuntimeError(f"Partial completion bitmap has the wrong size: {complete_path}")
        features = np.memmap(feature_path, mode="r+", dtype=np.uint16, shape=shape)
        completed = np.memmap(complete_path, mode="r+", dtype=np.uint8, shape=(len(image_dataset),))
    else:
        features = np.memmap(feature_path, mode="w+", dtype=np.uint16, shape=shape)
        completed = np.memmap(
            complete_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(image_dataset),),
        )
        completed[:] = 0
        completed.flush()
        _atomic_json(
            metadata_path,
            {
                **expected_metadata,
                "complete": False,
                "completed_images": 0,
                "created_at_unix": time.time(),
            },
        )

    missing_indices = np.flatnonzero(completed == 0).tolist()
    subset = torch.utils.data.Subset(image_dataset, missing_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    pending: list[int] = []
    started = time.time()
    for batch_number, (indices, images) in enumerate(loader, start=1):
        images = images.to(tower.device_name, non_blocking=True)
        with autocast_factory():
            tokens = tower.sequence_features(tower.normalize(images))
        if tuple(tokens.shape[1:]) != shape[1:]:
            raise RuntimeError(
                f"Cached feature shape mismatch: expected {shape[1:]}, got {tuple(tokens.shape[1:])}"
            )
        bits = tokens.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
        rows = indices.numpy().astype(np.int64)
        features[rows] = bits
        pending.extend(int(value) for value in rows)
        if batch_number % flush_interval == 0:
            features.flush()
            completed[pending] = 1
            completed.flush()
            pending.clear()
            done = int(completed.sum())
            elapsed = time.time() - started
            print(
                f"[feature-cache] {done}/{len(image_dataset)} images "
                f"({done / max(elapsed, 1e-6):.2f} images/s)",
                flush=True,
            )
    features.flush()
    if pending:
        completed[pending] = 1
    completed.flush()
    done = int(completed.sum())
    if done != len(image_dataset):
        raise RuntimeError(f"Feature cache is incomplete: {done}/{len(image_dataset)}")
    _atomic_json(
        metadata_path,
        {
            **expected_metadata,
            "complete": True,
            "completed_images": done,
            "completed_at_unix": time.time(),
        },
    )
    del features
    del completed
    return GroundingFeatureCache(cache_dir)
