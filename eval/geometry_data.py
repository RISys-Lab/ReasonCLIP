#!/usr/bin/env python3
"""Datasets and transforms for the TIPS/Probe3D geometry protocols.

The NAVI parsing, depth conversion, crop, and normal construction follow the
MIT-licensed Probe3D implementation by Mohamed El Banani.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

NYU_GEONET_BAD_SORTED_INDICES = (6_919, 21_181)
NYU_GEONET_BAD_FILES = (
    "bedroom_0129_r-1315328302.785180-1089784518.mat",
    "library_0006_r-1300708259.539496-3335480784.mat",
)
NYU_GEONET_ARCHIVE_SAMPLES = 30_916
NYU_GEONET_USABLE_SAMPLES = 30_914


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2 pickles in the NumPy 1.x evaluation environment."""

    def find_class(self, module: str, name: str):
        numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
        if numpy_major < 2 and (
            module == "numpy._core" or module.startswith("numpy._core.")
        ):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def clean_nyu_geonet_instances(instances: list[str]) -> tuple[list[str], bool]:
    precleaned = len(instances) == NYU_GEONET_USABLE_SAMPLES
    if len(instances) == NYU_GEONET_ARCHIVE_SAMPLES:
        observed_bad = tuple(instances[index] for index in NYU_GEONET_BAD_SORTED_INDICES)
        if observed_bad != NYU_GEONET_BAD_FILES:
            raise RuntimeError(
                f"Unexpected bad samples at the expected GeoNet indices: {observed_bad}"
            )
        instances = instances.copy()
        for index in reversed(NYU_GEONET_BAD_SORTED_INDICES):
            del instances[index]
    elif precleaned:
        present_bad = sorted(set(instances).intersection(NYU_GEONET_BAD_FILES))
        if present_bad:
            raise RuntimeError(f"Precleaned GeoNet root still contains {present_bad}")
    else:
        raise RuntimeError(
            "Expected 30,916 raw or 30,914 precleaned GeoNet files, "
            f"got {len(instances)}"
        )
    if len(instances) != NYU_GEONET_USABLE_SAMPLES:
        raise RuntimeError(f"Expected 30,914 usable GeoNet samples, got {len(instances)}")
    return instances, precleaned


def image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {array.shape}")
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255.0)


def maybe_color_jitter(image: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and random.random() < 0.8:
        image = ColorJitter(0.2, 0.2, 0.2, 0.2)(image)
    return image


def shared_random_resized_crop(
    image: torch.Tensor,
    depth: torch.Tensor,
    normals: torch.Tensor,
    output_size: tuple[int, int],
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled or random.random() >= 0.5:
        return image, depth, normals
    top, left, height, width = RandomResizedCrop.get_params(
        image,
        scale=(0.5, 1.0),
        ratio=(1.0, 1.0),
    )
    kwargs = {
        "interpolation": InterpolationMode.NEAREST,
        "antialias": False,
    }
    image = TF.resized_crop(image, top, left, height, width, output_size, **kwargs)
    depth = TF.resized_crop(depth, top, left, height, width, output_size, **kwargs)
    normals = TF.resized_crop(normals, top, left, height, width, output_size, **kwargs)
    return image, depth, normals


@dataclass(frozen=True)
class NYUDepthRecord:
    image_path: Path
    depth_path: Path
    focal_length: float
    sample_id: str


def nyu_depth_train_transform(
    image: np.ndarray,
    depth: np.ndarray,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply TIPS-resolution NYUv2 augmentation before normalization."""

    image = np.array(image, copy=True)
    depth = np.array(depth, dtype=np.float32, copy=True)
    if image.shape != (480, 640, 3) or depth.shape != (480, 640):
        raise ValueError(
            f"NYUv2 source must be 480x640; got {image.shape}, {depth.shape}"
        )

    if augment:
        rotate = np.random.rand() < 0.5
        degree = np.random.uniform(-2.5, 2.5)
        if rotate:
            height, width = depth.shape
            center = ((width - 1) * 0.5, (height - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(center, -degree, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            depth = cv2.warpAffine(
                depth,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            depth = np.flip(depth, axis=1)

    if augment and np.random.rand() < 0.5:
        gamma = np.random.uniform(0.9, 1.1)
        brightness = np.random.uniform(0.75, 1.25)
        colors = np.random.uniform(0.9, 1.1, size=3)
        image = image**gamma
        image = image * brightness
        image = image * colors.reshape(1, 1, 3)
        image = np.clip(image, 0, 255)

    return np.ascontiguousarray(image), np.ascontiguousarray(depth)


class NYUDepthDataset(Dataset):
    """BTS synchronized NYUv2 split evaluated with the TIPS resolution."""

    expected_samples = {"train": 24_231, "test": 654}
    max_depth = 10.0

    def __init__(
        self,
        root: Path,
        split: str,
        augment: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if split not in self.expected_samples:
            raise ValueError(f"Unsupported NYUv2 depth split: {split}")
        self.root = root
        self.split = split
        self.augment = augment and split == "train"
        self.split_path = root / f"nyu_{split}.txt"
        if not self.split_path.is_file():
            raise FileNotFoundError(self.split_path)
        split_bytes = self.split_path.read_bytes()
        self.split_sha256 = hashlib.sha256(split_bytes).hexdigest()

        records: list[NYUDepthRecord] = []
        for line_number, line in enumerate(split_bytes.decode("utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 3:
                raise RuntimeError(
                    f"Malformed {self.split_path}:{line_number}; expected RGB depth focal"
                )
            image_text, depth_text, focal_text = fields
            image_relative = Path(image_text.lstrip("/"))
            depth_relative = Path(depth_text.lstrip("/"))
            if ".." in image_relative.parts or ".." in depth_relative.parts:
                raise RuntimeError(f"Unsafe path in {self.split_path}:{line_number}")
            records.append(
                NYUDepthRecord(
                    image_path=root / image_relative,
                    depth_path=root / depth_relative,
                    focal_length=float(focal_text),
                    sample_id=image_relative.as_posix(),
                )
            )

        expected = self.expected_samples[split]
        if len(records) != expected:
            raise RuntimeError(
                f"Expected {expected:,} NYUv2 {split} records, got {len(records):,}"
            )
        if len({record.sample_id for record in records}) != len(records):
            raise RuntimeError(f"Duplicate RGB records in {self.split_path}")
        missing = [
            record
            for record in records
            if not record.image_path.is_file() or not record.depth_path.is_file()
        ]
        if missing:
            first = missing[0]
            raise RuntimeError(
                f"NYUv2 {split} has {len(missing):,} missing RGB/depth pairs; "
                f"first={first.image_path}, {first.depth_path}"
            )

        records.sort(key=lambda record: record.sample_id)
        self.total_samples = len(records)
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = np.array(source.convert("RGB"), copy=True)
        with Image.open(record.depth_path) as source:
            depth = np.array(source, dtype=np.float32, copy=True) / 1000.0
        if image.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise RuntimeError(
                f"Unexpected NYUv2 shape for {record.sample_id}: {image.shape}, {depth.shape}"
            )
        if self.split == "train":
            image, depth = nyu_depth_train_transform(image, depth, self.augment)
        return {
            "image": image_to_tensor(image),
            "depth": torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0),
            "focal_length": record.focal_length,
            "sample_id": record.sample_id,
        }


class NYUGeoNetDataset(Dataset):
    """The 30,914 usable GeoNet samples used by TIPS and Probe3D."""

    max_depth = 10.0

    def __init__(
        self,
        root: Path,
        task: str,
        augment: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.root = root
        if task not in {"depth", "normals"}:
            raise ValueError(f"Unsupported NYUv2 task: {task}")
        self.task = task
        self.augment = augment
        instances = sorted(path.name for path in root.glob("*.mat"))
        instances, self.precleaned = clean_nyu_geonet_instances(instances)
        self.total_samples = len(instances)
        if max_samples is not None:
            instances = instances[:max_samples]
        self.instances = instances
        self.center_crop = task == "normals"

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_name = self.instances[index]
        instance = scipy.io.loadmat(self.root / file_name)
        image = np.array(instance["img"][:480, :640], copy=True)
        depth = np.array(instance["depth"][:480, :640], dtype=np.float32, copy=True)
        normals = np.array(instance["norm"][:480, :640], dtype=np.float32, copy=True)

        image[:, :, 0] = image[:, :, 0] + 2 * 122.175
        image[:, :, 1] = image[:, :, 1] + 2 * 116.169
        image[:, :, 2] = image[:, :, 2] + 2 * 103.508
        image = image.astype(np.uint8)
        depth[depth > self.max_depth] = 0

        image_t = maybe_color_jitter(image_to_tensor(image), self.augment)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        normals_t = torch.from_numpy(normals).permute(2, 0, 1)
        if self.center_crop:
            image_t = image_t[..., 80:-80]
            depth_t = depth_t[..., 80:-80]
            normals_t = normals_t[..., 80:-80]
            output_size = (480, 480)
        else:
            output_size = (480, 640)

        image_t, depth_t, normals_t = shared_random_resized_crop(
            image_t,
            depth_t,
            normals_t,
            output_size,
            self.augment,
        )
        return {
            "image": image_t,
            "depth": depth_t,
            "normals": normals_t,
            "sample_id": file_name,
        }


class NYUTestDataset(Dataset):
    """NYUv2 labeled test split with FAIR Ladicky normal metadata."""

    max_depth = 10.0

    def __init__(self, pickle_path: Path, max_samples: int | None = None) -> None:
        with pickle_path.open("rb") as handle:
            self.data = NumpyCompatUnpickler(handle).load()
        required = {"test_indices", "depths", "images", "snorms"}
        missing = required.difference(self.data)
        if missing:
            raise RuntimeError(f"NYUv2 test pickle lacks keys: {sorted(missing)}")
        self.indices = list(self.data["test_indices"])
        if len(self.indices) != 654:
            raise RuntimeError(f"Expected 654 NYUv2 test samples, got {len(self.indices)}")
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        nyu_index = int(self.indices[index])
        image = np.asarray(self.data["images"][nyu_index])
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        depth = np.array(self.data["depths"][nyu_index], dtype=np.float32, copy=True)
        normals = np.array(self.data["snorms"][nyu_index], dtype=np.float32, copy=True)
        if normals.shape[-1] == 3:
            normals = np.transpose(normals, (2, 0, 1))
        depth[depth > self.max_depth] = 0
        return {
            "image": image_to_tensor(image),
            "depth": torch.from_numpy(depth).unsqueeze(0),
            "normals": torch.from_numpy(normals),
            "sample_id": f"nyu_{nyu_index:04d}",
        }


def read_navi_image(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image_to_tensor(image)


def read_navi_depth(path: Path, scale_factor: float = 10.0) -> torch.Tensor:
    with Image.open(path) as source:
        disparity = np.asarray(source).astype("uint16").astype(np.float32)
    disparity /= ((2**16) - 1) * scale_factor
    disparity[disparity == 0] = np.inf
    depth = 1.0 / disparity
    return torch.from_numpy(depth).unsqueeze(0).div_(1000.0)


def normalize_navi_relative_depth(
    depth: torch.Tensor,
    min_depth: torch.Tensor,
) -> torch.Tensor:
    """Apply Probe3D's per-image NAVI relative-depth normalization."""

    valid = depth > 0
    if not valid.any():
        raise RuntimeError("NAVI crop contains no valid depth pixels")
    denominator = (depth.max() - min_depth).clamp_min(0.01)
    normalized = (depth - min_depth) / denominator
    normalized = normalized * 0.99 + 0.01
    return normalized.masked_fill(~valid, 0)


def resize_short_and_center_crop(tensor: torch.Tensor, size: int = 512) -> torch.Tensor:
    tensor = TF.resize(
        tensor,
        size,
        interpolation=InterpolationMode.NEAREST,
        antialias=False,
    )
    return TF.center_crop(tensor, [size, size])


def bbox_crop(image: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_coords = (depth > 0).nonzero()
    if mask_coords.numel() == 0:
        raise RuntimeError("NAVI depth map contains no valid pixels")
    top_left = mask_coords.min(dim=0).values[1:]
    bottom_right = mask_coords.max(dim=0).values[1:]
    box_size = bottom_right - top_left
    image_size = torch.tensor(depth.shape[1:])
    if box_size.max() > image_size.min():
        raise RuntimeError("NAVI target aspect ratio prevents a square crop")

    pad_size = box_size.max() - box_size
    centered_top_left = top_left - pad_size // 2
    centered_bottom_right = centered_top_left + box_size.max()
    if (centered_top_left >= 0).all() and (centered_bottom_right <= image_size).all():
        crop_top_left = centered_top_left
        crop_bottom_right = centered_bottom_right
    else:
        crop_top_left = (top_left - pad_size).clip(min=0)
        crop_bottom_right = crop_top_left + box_size.max()

    y0, x0 = (int(value) for value in crop_top_left)
    y1, x1 = (int(value) for value in crop_bottom_right)
    return image[:, y0:y1, x0:x1], depth[:, y0:y1, x0:x1]


def coordinate_grid(height: int, width: int) -> torch.Tensor:
    grid_x = torch.linspace(0.5, width - 0.5, width)
    grid_y = torch.linspace(0.5, height - 0.5, height)
    xs = grid_x.view(1, width).repeat(height, 1)
    ys = grid_y.view(height, 1).repeat(1, width)
    return torch.stack((xs, ys, torch.ones_like(xs)), dim=0)


def depth_to_normals(depth: torch.Tensor, focal_length: float) -> torch.Tensor:
    valid = (depth > 0).float()
    filled = depth.clone()
    filled[filled == 0] = 1e6
    _, height, width = filled.shape
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    xyd = coordinate_grid(height, width) * filled
    xyz = (torch.inverse(intrinsics) @ xyd.view(3, -1)).view(3, height, width)

    diff_left = xyz[:, 1:-1, :-2] - xyz[:, 1:-1, 1:-1]
    diff_top = xyz[:, :-2, 1:-1] - xyz[:, 1:-1, 1:-1]
    diff_right = xyz[:, 1:-1, 2:] - xyz[:, 1:-1, 1:-1]
    diff_bottom = xyz[:, 2:, 1:-1] - xyz[:, 1:-1, 1:-1]
    normal = torch.zeros_like(xyz)
    normal[:, 1:-1, 1:-1] = (
        torch.linalg.cross(diff_left, diff_top, dim=0)
        + torch.linalg.cross(diff_top, diff_right, dim=0)
        + torch.linalg.cross(diff_right, diff_bottom, dim=0)
        + torch.linalg.cross(diff_bottom, diff_left, dim=0)
    ) / 4.0
    return F.normalize(normal, p=2, dim=0) * valid


class NAVIProbeDataset(Dataset):
    """Probe3D NAVI trainval/test splits and object-centric transforms."""

    max_depth = 1.0

    def __init__(
        self,
        root: Path,
        split: str,
        augment: bool = True,
        relative_depth: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"trainval", "test"}:
            raise ValueError(f"Unsupported NAVI split: {split}")
        self.root = root
        self.split = split
        self.augment = augment and split == "trainval"
        self.relative_depth = relative_depth
        self.data = self._parse_dataset()
        self.instances = self._build_instances()
        self.raw_samples = len(self.instances)
        self.eligible_objects = sorted({instance[0] for instance in self.instances})
        self.instances = self.instances[::4]
        self.total_samples = len(self.instances)
        if max_samples is not None:
            self.instances = self.instances[:max_samples]

    def _parse_dataset(self) -> dict[str, dict[str, dict[str, Any]]]:
        collections = glob.glob(str(self.root / "*" / "multiview_*"))
        collections += glob.glob(str(self.root / "*" / "wild_set"))
        if not collections:
            incompatible = list(self.root.glob("*/multiview-*"))
            raise RuntimeError(
                f"No Probe3D NAVI scenes under {self.root}; "
                f"found {len(incompatible)} incompatible multiview-* scenes"
            )
        data: dict[str, dict[str, dict[str, Any]]] = {}
        for collection_path_text in collections:
            collection_path = Path(collection_path_text)
            object_id = collection_path.parent.name
            collection_id = collection_path.name
            image_ids = [
                Path(name).stem
                for name in os.listdir(collection_path / "images")
                if name.lower().endswith(".jpg") and "_" not in Path(name).stem
            ]
            with (collection_path / "annotations.json").open("r", encoding="utf-8") as handle:
                annotation_list = json.load(handle)
            annotations = {Path(item["filename"]).stem: item for item in annotation_list}
            data.setdefault(object_id, {})[collection_id] = {
                "views": image_ids,
                "annotations": annotations,
            }
        return data

    def _build_instances(self) -> list[tuple[str, str, str]]:
        collection = "multiview" if self.split == "trainval" else "wild"
        instances: list[tuple[str, str, str]] = []
        for object_id, object_data in self.data.items():
            scenes = list(object_data.keys())
            if "wild_set" not in scenes or len(scenes) == 1:
                continue
            if collection == "wild":
                for image_id in object_data["wild_set"]["views"]:
                    instances.append((object_id, "wild_set", image_id))
            else:
                for scene_id in scenes:
                    if "multiview" not in scene_id:
                        continue
                    for image_id in object_data[scene_id]["views"]:
                        instances.append((object_id, scene_id, image_id))
        return instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        object_id, scene_id, image_id = self.instances[index]
        scene = self.root / object_id / scene_id
        annotation = self.data[object_id][scene_id]["annotations"][image_id]
        image = read_navi_image(scene / "images" / f"downsampled_{image_id}.jpg")
        depth = read_navi_depth(scene / "depth" / f"downsampled_{image_id}.png")
        min_depth = depth[depth > 0].min()

        image = resize_short_and_center_crop(image)
        image = maybe_color_jitter(image, self.augment)
        depth = resize_short_and_center_crop(depth)
        original_height, original_width = annotation["image_size"]
        image_height, image_width = image.shape[1:]
        original_focal = float(annotation["camera"]["focal_length"])
        augmented_focal = original_focal * min(image_height, image_width) / min(
            original_height,
            original_width,
        )

        image, depth = bbox_crop(image, depth)
        normals = depth_to_normals(depth.clone(), augmented_focal)
        image = F.interpolate(image.unsqueeze(0), (512, 512), mode="nearest").squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), (512, 512), mode="nearest").squeeze(0)
        normals = F.interpolate(normals.unsqueeze(0), (512, 512), mode="nearest").squeeze(0)
        depth[depth < min_depth] = 0
        if self.relative_depth:
            depth = normalize_navi_relative_depth(depth, min_depth)
        return {
            "image": image,
            "depth": depth,
            "normals": normals,
            "sample_id": f"{object_id}/{scene_id}/{image_id}",
        }


def dataset_protocol(dataset: Dataset) -> dict[str, Any]:
    if isinstance(dataset, NYUDepthDataset):
        return {
            "name": "NYUv2-BTS-sync",
            "split": dataset.split,
            "samples": len(dataset),
            "full_samples": dataset.total_samples,
            "source_resolution": [480, 640],
            "resolution": [480, 640],
            "augment": dataset.augment,
            "split_file": dataset.split_path.name,
            "split_sha256": dataset.split_sha256,
            "depth_scale": 1000,
            "pipeline": (
                "full-resolution 480x640 -> RandomRotate(2.5,p=0.5) -> "
                "RandomFlip(p=0.5) -> ColorAug(p=0.5)"
                if dataset.split == "train"
                else "full-resolution 480x640"
            ),
        }
    if isinstance(dataset, NYUGeoNetDataset):
        return {
            "name": "NYU-GeoNet",
            "task": dataset.task,
            "split": "trainval",
            "samples": len(dataset),
            "full_samples": dataset.total_samples,
            "source_resolution": [480, 640],
            "resolution": [480, 480] if dataset.center_crop else [480, 640],
            "center_crop": dataset.center_crop,
            "augment": dataset.augment,
            "augmentation": (
                "ColorJitter(p=0.8) + RandomResizedCrop(scale=0.5-1.0,p=0.5); "
                "no rotation or horizontal flip"
                if dataset.augment
                else None
            ),
            "normal_source": "GeoNet extracted NYUv2 surface normals",
            "valid_mask": "metric depth > 0 after removing depth > 10m",
            "archive_samples": NYU_GEONET_ARCHIVE_SAMPLES,
            "precleaned_root": dataset.precleaned,
            "removed_sorted_indices": list(NYU_GEONET_BAD_SORTED_INDICES),
            "removed_files": list(NYU_GEONET_BAD_FILES),
        }
    if isinstance(dataset, NYUTestDataset):
        return {
            "name": "NYUv2-labeled",
            "task": "normals",
            "split": "test",
            "samples": len(dataset),
            "resolution": [480, 640],
            "normal_source": "Ladicky surface-normal metadata",
            "valid_mask": "metric depth > 0 after removing depth > 10m",
        }
    if isinstance(dataset, NAVIProbeDataset):
        return {
            "name": "NAVI",
            "split": dataset.split,
            "samples": len(dataset),
            "full_samples": dataset.total_samples,
            "samples_before_stride": dataset.raw_samples,
            "eligible_objects": len(dataset.eligible_objects),
            "resolution": [512, 512],
            "stride": 4,
            "relative_depth": dataset.relative_depth,
            "relative_depth_valid_range": [0.01, 1.0] if dataset.relative_depth else None,
            "augment": dataset.augment,
            "collection_pattern": "multiview_*" if dataset.split == "trainval" else "wild_set",
            "normal_source": "Probe3D depth-to-normal construction",
            "normal_coordinate_frame": "+x right, +y down, +z into camera",
            "normal_valid_mask": "metric depth > 0",
        }
    raise TypeError(type(dataset))
