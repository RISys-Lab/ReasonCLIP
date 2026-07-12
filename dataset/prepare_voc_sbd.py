#!/usr/bin/env python3
"""Prepare the official VOC2012 + SBD 10,582-image segmentation split."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVKIT = REPO_ROOT / "rebuttal" / "downstream_data" / "VOCdevkit"
EXPECTED_TRAIN = 1464
EXPECTED_AUG = 9118
EXPECTED_TRAINAUG = 10_582
EXPECTED_VAL = 1449


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devkit-root", type=Path, default=DEFAULT_DEVKIT)
    parser.add_argument("--sbd-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_mask(job: tuple[Path, Path, bool]) -> tuple[str, bool]:
    source, destination, overwrite = job
    if destination.is_file() and not overwrite:
        return source.stem, False
    payload = loadmat(source)
    mask = payload["GTcls"][0]["Segmentation"][0].astype(np.uint8)
    labels = set(np.unique(mask).tolist())
    if any(label not in set(range(21)) | {255} for label in labels):
        raise ValueError(f"Unexpected labels in {source}: {sorted(labels)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    Image.fromarray(mask).save(temporary, format="PNG")
    temporary.replace(destination)
    return source.stem, True


def main() -> None:
    args = parse_args()
    voc_root = args.devkit_root / "VOC2012"
    sbd_root = args.sbd_root or args.devkit_root / "VOCaug" / "benchmark_RELEASE"
    sbd_dataset = sbd_root / "dataset"
    split_root = voc_root / "ImageSets" / "Segmentation"
    output_masks = voc_root / "SegmentationClassAug"

    voc_train = read_ids(split_root / "train.txt")
    voc_val = read_ids(split_root / "val.txt")
    sbd_train = read_ids(sbd_dataset / "train.txt")
    sbd_val = read_ids(sbd_dataset / "val.txt")
    full_sbd = set(sbd_train) | set(sbd_val)
    trainaug = sorted(set(voc_train) | full_sbd - set(voc_val))
    aug = sorted(full_sbd - set(voc_train) - set(voc_val))

    if len(voc_train) != EXPECTED_TRAIN or len(voc_val) != EXPECTED_VAL:
        raise RuntimeError(
            f"VOC split mismatch: train={len(voc_train)} val={len(voc_val)}; "
            f"expected {EXPECTED_TRAIN}/{EXPECTED_VAL}"
        )
    if len(trainaug) != EXPECTED_TRAINAUG or len(aug) != EXPECTED_AUG:
        raise RuntimeError(
            f"SBD merge mismatch: trainaug={len(trainaug)} aug={len(aug)}; "
            f"expected {EXPECTED_TRAINAUG}/{EXPECTED_AUG}"
        )
    if set(trainaug) & set(voc_val):
        raise RuntimeError("VOC validation images leaked into trainaug")

    mat_files = sorted((sbd_dataset / "cls").glob("*.mat"))
    if len(mat_files) != 11_355:
        raise RuntimeError(f"Expected 11,355 SBD masks, found {len(mat_files)}")
    jobs = [(path, output_masks / f"{path.stem}.png", args.overwrite) for path in mat_files]
    converted = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for _, changed in tqdm(executor.map(convert_mask, jobs), total=len(jobs), desc="SBD masks"):
                converted += int(changed)
    else:
        for job in tqdm(jobs, desc="SBD masks"):
            _, changed = convert_mask(job)
            converted += int(changed)

    (split_root / "trainaug.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in trainaug), encoding="utf-8"
    )
    (split_root / "aug.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in aug), encoding="utf-8"
    )

    sbd_images = sbd_dataset / "img"
    missing_images = [sample_id for sample_id in aug if not (sbd_images / f"{sample_id}.jpg").is_file()]
    missing_masks = [sample_id for sample_id in aug if not (output_masks / f"{sample_id}.png").is_file()]
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f"Prepared SBD data incomplete: missing_images={len(missing_images)} "
            f"missing_masks={len(missing_masks)}"
        )

    manifest = {
        "voc_train": len(voc_train),
        "sbd_additional": len(aug),
        "trainaug": len(trainaug),
        "voc_val": len(voc_val),
        "sbd_masks_total": len(mat_files),
        "masks_converted_this_run": converted,
        "validation_leakage": 0,
        "sbd_root": str(sbd_root),
        "mask_output": str(output_masks),
    }
    manifest_path = args.devkit_root / "voc_sbd_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
