#!/usr/bin/env python3
"""Prepare and verify NAVI exactly as expected by Probe3D geometry probes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from tqdm import tqdm


EXPECTED_ARCHIVE_BYTES = 31_098_738_677
EXPECTED_ARCHIVE_MD5 = "0868b56af7747f2bf766d4ed2081f853"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-archive-hash", action="store_true")
    return parser.parse_args()


def file_md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resized_path(path: Path) -> Path:
    return path.with_name(f"downsampled_{path.name}")


def resize_image(job: tuple[str, int, int, bool]) -> str:
    path_text, resample, min_size, overwrite = job
    path = Path(path_text)
    destination = resized_path(path)
    if destination.is_file() and not overwrite:
        return "existing"

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        if min(width, height) <= 0:
            raise ValueError(f"Invalid image size for {path}: {image.size}")
        factor = float(min_size) / min(width, height)
        target_size = (int(width * factor), int(height * factor))
        image = image.resize(target_size, resample)
        temporary = destination.with_name(f".{destination.name}.tmp")
        image.save(temporary, format=source.format)
    os.replace(temporary, destination)
    return "written"


def source_files(root: Path, subdir: str, suffix: str) -> list[Path]:
    pattern = str(root / "*" / "*" / subdir / f"*.{suffix}")
    return [
        Path(path)
        for path in glob.glob(pattern)
        if not Path(path).name.startswith("downsampled_")
    ]


def run_resize(
    paths: list[Path],
    resample: Image.Resampling,
    min_size: int,
    workers: int,
    overwrite: bool,
    label: str,
) -> dict[str, int]:
    jobs = [(str(path), int(resample), min_size, overwrite) for path in paths]
    counts = {"written": 0, "existing": 0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(resize_image, jobs, chunksize=8)
        for status in tqdm(results, total=len(jobs), desc=label):
            counts[status] += 1
    return counts


def scene_paths(root: Path) -> tuple[list[Path], list[Path]]:
    multiview = [Path(path) for path in glob.glob(str(root / "*" / "multiview_*"))]
    wild = [Path(path) for path in glob.glob(str(root / "*" / "wild_set"))]
    if not multiview:
        incompatible = list(root.glob("*/multiview-*"))
        suffix = f"; found {len(incompatible)} incompatible multiview-* scenes" if incompatible else ""
        raise RuntimeError(f"No Probe3D-compatible multiview_* scenes under {root}{suffix}")
    if not wild:
        raise RuntimeError(f"No wild_set scenes under {root}")
    return multiview, wild


def load_scene(scene: Path) -> tuple[list[str], dict[str, Any]]:
    annotation_path = scene / "annotations.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    annotations = {Path(record["filename"]).stem: record for record in records}

    image_dir = scene / "images"
    image_ids = [
        Path(name).stem
        for name in os.listdir(image_dir)
        if name.lower().endswith(".jpg") and "_" not in Path(name).stem
    ]
    missing_annotations = [image_id for image_id in image_ids if image_id not in annotations]
    if missing_annotations:
        raise RuntimeError(f"{scene}: {len(missing_annotations)} images lack annotations")
    return image_ids, annotations


def instance_digest(items: list[tuple[str, str, str]]) -> str:
    payload = "\n".join("/".join(item) for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_dataset(root: Path, min_size: int) -> dict[str, Any]:
    multiview, wild = scene_paths(root)
    ordered_scenes = multiview + wild
    scene_data: dict[str, dict[str, tuple[list[str], dict[str, Any]]]] = {}
    missing_resized: list[str] = []
    invalid_resized: list[str] = []

    for scene in tqdm(ordered_scenes, desc="verify scenes"):
        object_id = scene.parent.name
        image_ids, annotations = load_scene(scene)
        scene_data.setdefault(object_id, {})[scene.name] = (image_ids, annotations)
        for image_id in image_ids:
            image_path = scene / "images" / f"downsampled_{image_id}.jpg"
            depth_path = scene / "depth" / f"downsampled_{image_id}.png"
            if not image_path.is_file() or not depth_path.is_file():
                missing_resized.append(f"{object_id}/{scene.name}/{image_id}")
                continue
            with Image.open(image_path) as image, Image.open(depth_path) as depth:
                image_size = image.size
                depth_size = depth.size
            if image_size != depth_size or min(image_size) != min_size:
                invalid_resized.append(
                    f"{object_id}/{scene.name}/{image_id}: rgb={image_size}, depth={depth_size}"
                )

    if missing_resized:
        preview = ", ".join(missing_resized[:5])
        raise RuntimeError(f"Missing {len(missing_resized)} resized RGB/depth pairs: {preview}")
    if invalid_resized:
        preview = ", ".join(invalid_resized[:5])
        raise RuntimeError(f"Invalid {len(invalid_resized)} resized RGB/depth pairs: {preview}")

    objects = []
    train_instances: list[tuple[str, str, str]] = []
    test_instances: list[tuple[str, str, str]] = []
    for object_id, scenes in scene_data.items():
        if "wild_set" not in scenes or len(scenes) == 1:
            continue
        objects.append(object_id)
        for scene_id, (image_ids, _) in scenes.items():
            destination = test_instances if scene_id == "wild_set" else train_instances
            destination.extend((object_id, scene_id, image_id) for image_id in image_ids)

    train_sampled = train_instances[::4]
    test_sampled = test_instances[::4]
    return {
        "data_root": str(root.resolve()),
        "collection_glob": "*/multiview_* and */wild_set",
        "source_traversal": "glob.glob/os.listdir order, matching Probe3D",
        "minimum_resized_side": min_size,
        "object_count": len(objects),
        "multiview_scene_count": len(multiview),
        "wild_scene_count": len(wild),
        "trainval_instances_before_stride": len(train_instances),
        "trainval_instances_after_stride_4": len(train_sampled),
        "trainval_stride_4_sha256": instance_digest(train_sampled),
        "test_instances_before_stride": len(test_instances),
        "test_instances_after_stride_4": len(test_sampled),
        "test_stride_4_sha256": instance_digest(test_sampled),
        "missing_resized_pairs": 0,
        "invalid_resized_pairs": 0,
    }


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.min_size <= 0 or args.workers <= 0:
        raise ValueError("--min-size and --workers must be positive")
    scene_paths(root)

    archive_summary: dict[str, Any] | None = None
    if args.archive is not None:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_summary = {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "expected_bytes": EXPECTED_ARCHIVE_BYTES,
            "expected_md5": EXPECTED_ARCHIVE_MD5,
        }
        if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError(
                f"NAVI archive size mismatch: {archive.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
            )
        if not args.skip_archive_hash:
            archive_summary["md5"] = file_md5(archive)
            if archive_summary["md5"] != EXPECTED_ARCHIVE_MD5:
                raise RuntimeError(
                    f"NAVI archive MD5 mismatch: {archive_summary['md5']} != {EXPECTED_ARCHIVE_MD5}"
                )

    resize_summary: dict[str, Any] = {"verify_only": args.verify_only}
    if not args.verify_only:
        rgb_paths = source_files(root, "images", "jpg")
        depth_paths = source_files(root, "depth", "png")
        if not rgb_paths or not depth_paths:
            raise RuntimeError(f"No NAVI RGB/depth source images found under {root}")
        resize_summary.update(
            {
                "rgb_sources": len(rgb_paths),
                "depth_sources": len(depth_paths),
                "rgb": run_resize(
                    rgb_paths,
                    Image.Resampling.BICUBIC,
                    args.min_size,
                    args.workers,
                    args.overwrite,
                    "resize RGB",
                ),
                "depth": run_resize(
                    depth_paths,
                    Image.Resampling.NEAREST,
                    args.min_size,
                    args.workers,
                    args.overwrite,
                    "resize depth",
                ),
            }
        )

    summary = verify_dataset(root, args.min_size)
    summary["archive"] = archive_summary
    summary["resize"] = resize_summary
    manifest = args.manifest or root.parent / "probe3d_navi_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
