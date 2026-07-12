#!/usr/bin/env python3
"""Dataset utilities for the LocCa/SigLIP2 RefCOCO probe."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from official_grounding import (
    C4Tokenizer,
    TASK_PREFIX,
    box_string,
    prompt_string,
    quantize_box_xywh,
    target_string,
)


IMAGE_TRANSFORM_NAME = "tf_bilinear_uint8_no_antialias_v1"


@dataclass(frozen=True)
class GroundingRecord:
    dataset: str
    split: str
    split_by: str
    ref_id: int
    sentence_id: int
    ann_id: int
    image_id: int
    file_name: str
    width: int
    height: int
    bbox_xywh: tuple[float, float, float, float]
    expression: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundingRecord":
        values = dict(payload)
        values["bbox_xywh"] = tuple(float(value) for value in values["bbox_xywh"])
        for key in ("ref_id", "sentence_id", "ann_id", "image_id", "width", "height"):
            values[key] = int(values[key])
        return cls(**values)


@dataclass(frozen=True)
class GroundingImageGroup:
    image_id: int
    annotations: tuple[tuple[GroundingRecord, ...], ...]


def _sample_value(seed: int, count: int, image_id: int, salt: int) -> int:
    """Deterministic SplitMix64 value, stable across workers and resumes."""

    if count < 0:
        raise ValueError(f"Choice count must be non-negative, got {count}")
    mask = (1 << 64) - 1
    value = int(seed) & mask
    value ^= ((int(count) + 1) * 0xD1B54A32D192ED03) & mask
    value ^= (int(image_id) * 0x94D049BB133111EB) & mask
    value ^= int(salt) & mask
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return int(value ^ (value >> 31))


def _sample_offset(
    seed: int,
    count: int,
    image_id: int,
    salt: int,
    size: int,
) -> int:
    if size <= 0:
        raise ValueError(f"Choice size must be positive, got {size}")
    return _sample_value(seed, count, image_id, salt) % size


def _without_replacement_offset(
    seed: int,
    count: int,
    image_id: int,
    size: int,
) -> int:
    """Cycle through a fresh deterministic permutation before repeating."""

    if size <= 0:
        raise ValueError(f"Choice size must be positive, got {size}")
    cycle, offset = divmod(count, size)
    permutation = sorted(
        range(size),
        key=lambda index: (
            _sample_value(
                seed,
                cycle,
                image_id,
                0xA0761D6478BD642F ^ index,
            ),
            index,
        ),
    )
    return permutation[offset]


class EpochGroundingRecordSampler:
    """One image per epoch with Big Vision's two-stage RefCOCO choice."""

    def __init__(
        self,
        records: Sequence[GroundingRecord],
        *,
        sample_one_per_image: bool = False,
        seed: int = 0,
        max_images: int | None = None,
    ) -> None:
        if not records:
            raise ValueError("Grounding record sampler requires at least one record")
        self.sample_one_per_image = bool(sample_one_per_image)
        self.seed = int(seed)
        self.epoch = 0

        if not self.sample_one_per_image:
            selected = list(records[:max_images] if max_images is not None else records)
            if not selected:
                raise ValueError("Grounding record limit selected no records")
            self.records = selected
            self.image_groups: tuple[GroundingImageGroup, ...] = ()
            return

        by_image: dict[int, dict[int, list[GroundingRecord]]] = {}
        image_metadata: dict[int, tuple[str, int, int]] = {}
        for record in records:
            metadata = (record.file_name, record.width, record.height)
            previous = image_metadata.setdefault(record.image_id, metadata)
            if previous != metadata:
                raise RuntimeError(
                    f"Image ID {record.image_id} has conflicting metadata: "
                    f"{previous!r} != {metadata!r}"
                )
            by_image.setdefault(record.image_id, {}).setdefault(record.ann_id, []).append(record)

        image_ids = sorted(by_image)
        if max_images is not None:
            if max_images <= 0:
                raise ValueError(f"max_images must be positive, got {max_images}")
            image_ids = image_ids[:max_images]
        if not image_ids:
            raise ValueError("Grounding image limit selected no images")

        groups = []
        selected_records = []
        for image_id in image_ids:
            annotations = []
            for ann_id in sorted(by_image[image_id]):
                sentences = tuple(
                    sorted(
                        by_image[image_id][ann_id],
                        key=lambda record: (
                            record.dataset,
                            record.ref_id,
                            record.sentence_id,
                            record.expression,
                        ),
                    )
                )
                annotations.append(sentences)
                selected_records.extend(sentences)
            groups.append(GroundingImageGroup(image_id, tuple(annotations)))
        self.records = selected_records
        self.image_groups = tuple(groups)

    def __len__(self) -> int:
        return len(self.image_groups) if self.sample_one_per_image else len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")
        self.epoch = int(epoch)

    def record_at(self, index: int) -> GroundingRecord:
        if not self.sample_one_per_image:
            return self.records[index]
        group = self.image_groups[index]
        annotation_index = _without_replacement_offset(
            self.seed,
            self.epoch,
            group.image_id,
            len(group.annotations),
        )
        annotation = group.annotations[annotation_index]
        sentence_index = _sample_offset(
            self.seed,
            self.epoch,
            group.image_id,
            0xE7037ED1A0B428DB ^ annotation[0].ann_id,
            len(annotation),
        )
        return annotation[sentence_index]

    def records_for_epoch(self, epoch: int | None = None) -> list[GroundingRecord]:
        previous = self.epoch
        if epoch is not None:
            self.set_epoch(epoch)
        try:
            return [self.record_at(index) for index in range(len(self))]
        finally:
            self.epoch = previous

    def summary(self) -> dict[str, Any]:
        return {
            "unit": "unique_image" if self.sample_one_per_image else "referring_sentence",
            "examples_per_epoch": len(self),
            "candidate_sentences": len(self.records),
            "candidate_annotations": len(
                {(record.image_id, record.ann_id) for record in self.records}
            ),
            "unique_images": len({record.image_id for record in self.records}),
            "selection": (
                "shuffled no-replacement annotation cycle per image, then uniform "
                "sentence with replacement"
                if self.sample_one_per_image
                else "all referring sentences"
            ),
        }


def load_grounding_records(path: Path, max_samples: int | None = None) -> list[GroundingRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(GroundingRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise RuntimeError(f"Invalid record at {path}:{line_number}") from exc
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise RuntimeError(f"No grounding records in {path}")
    return records


def image_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    pixels = torch.from_numpy(
        np.array(image, dtype=np.uint8, copy=True)
    ).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        pixels,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    # Big Vision resize casts the floating result back to the input uint8 dtype.
    return resized.squeeze(0).clamp_(0, 255).to(torch.uint8).float().div_(255.0)


class RefCOCOLocCaDataset(Dataset):
    """Direct-resize RefCOCO examples for frozen-encoder decoder training."""

    def __init__(
        self,
        records_path: Path,
        image_root: Path,
        resolution: int,
        max_samples: int | None = None,
        verify_images: bool = True,
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
        self.image_root = image_root
        self.resolution = int(resolution)
        if self.resolution <= 0:
            raise ValueError(f"Invalid resolution: {resolution}")
        if verify_images:
            missing = sorted(
                {
                    record.file_name
                    for record in self.records
                    if not (self.image_root / record.file_name).is_file()
                }
            )
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing {len(missing)} RefCOCO images under {self.image_root}; first: {preview}"
                )

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
        with Image.open(self.image_root / record.file_name) as source:
            image = image_to_tensor(source, self.resolution)
        return {
            "image": image,
            "record": record,
            "box_lbrt": quantize_box_xywh(
                record.bbox_xywh,
                record.width,
                record.height,
            ),
        }


def collate_grounding(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in items]),
        "records": [item["record"] for item in items],
        "boxes_lbrt": [item["box_lbrt"] for item in items],
    }


def training_tokens(
    records: Sequence[GroundingRecord],
    boxes_lbrt: Sequence[Sequence[int]],
    tokenizer: C4Tokenizer,
    loss_scope: str = "box_suffix",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if len(records) != len(boxes_lbrt):
        raise ValueError("Grounding records and boxes have different batch sizes")
    if loss_scope not in {"box_suffix", "full_aref"}:
        raise ValueError(f"Unsupported grounding loss scope: {loss_scope}")
    token_rows = []
    mask_rows = []
    truncated_boxes = 0
    for record, box in zip(records, boxes_lbrt):
        prompt_ids = tokenizer.encode(prompt_string(record.expression), add_eos=False)
        task_prefix_ids = tokenizer.encode(TASK_PREFIX, add_eos=False)
        text = target_string(record.expression, box)
        untruncated = tokenizer.encode(text, add_eos=True)
        if len(untruncated) > tokenizer.max_length:
            box_ids = tokenizer.encode(box_string(box), add_eos=False)
            retained = untruncated[: tokenizer.max_length - 1]
            if len(box_ids) > len(retained) or retained[-len(box_ids) :] != box_ids:
                truncated_boxes += 1
        encoded = tokenizer.encode_sticky(text)
        retained_prompt_length = min(
            len(prompt_ids),
            (
                tokenizer.max_length - 1
                if len(untruncated) >= tokenizer.max_length
                else len(prompt_ids)
            ),
        )
        if encoded[:retained_prompt_length] != prompt_ids[:retained_prompt_length]:
            raise RuntimeError("SentencePiece expression-prompt tokenization changed in context")
        if encoded[: len(task_prefix_ids)] != task_prefix_ids:
            raise RuntimeError("SentencePiece ARef task-prefix tokenization changed in context")
        token_rows.append(encoded)
        row_mask = [token_id != tokenizer.pad_id for token_id in encoded]
        # The downstream REC probe conditions on the expression and predicts the
        # box. full_aref preserves LocCa's distinct pretraining objective.
        loss_start = (
            retained_prompt_length
            if loss_scope == "box_suffix"
            else len(task_prefix_ids)
        )
        row_mask[:loss_start] = [False] * loss_start
        mask_rows.append(row_mask)
    labels = torch.tensor(token_rows, dtype=torch.long)
    loss_mask = torch.tensor(mask_rows, dtype=torch.bool)
    return labels, loss_mask, truncated_boxes


def prompt_tokens(
    records: Sequence[GroundingRecord],
    tokenizer: C4Tokenizer,
) -> tuple[list[list[int]], int]:
    prompts = []
    truncated = 0
    for record in records:
        ids = tokenizer.encode(prompt_string(record.expression), add_eos=False)
        if len(ids) >= tokenizer.max_length:
            ids = ids[: tokenizer.max_length - 1]
            truncated += 1
        prompts.append(ids)
    return prompts, truncated
