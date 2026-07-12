#!/usr/bin/env python3
"""Prepare NYUv2 normal-estimation artifacts used by Probe3D and TIPS."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


LABELED_MAT_BYTES = 2_972_037_809
NORMALS_ARCHIVE_BYTES = 841_839_365
NORMALS_ARCHIVE_MD5 = "e51209f4960b73c014d5fdeccc24b669"
GEONET_ARCHIVE_BYTES = {
    "data1.zip": 65_806_938_119,
    "data2.zip": 75_320_754_560,
}
GEONET_BAD_SORTED_INDICES = (6_919, 21_181)
GEONET_BAD_FILES = (
    "bedroom_0129_r-1315328302.785180-1089784518.mat",
    "library_0006_r-1300708259.539496-3335480784.mat",
)
GEONET_ARCHIVE_SAMPLES = 30_916
GEONET_USABLE_SAMPLES = 30_914


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nyu-root",
        type=Path,
        default=Path("rebuttal/downstream_data/NYUv2"),
    )
    parser.add_argument("--labeled-mat", type=Path)
    parser.add_argument("--normals-archive", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--test-pickle", type=Path)
    parser.add_argument("--geonet-root", type=Path)
    parser.add_argument("--geonet-archive", type=Path, action="append", default=[])
    parser.add_argument("--skip-test-pickle", action="store_true")
    parser.add_argument("--skip-geonet", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def file_md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_md5: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(f"File size mismatch for {path}: {size} != {expected_bytes}")
    summary: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": size,
        "expected_bytes": expected_bytes,
    }
    if expected_md5 is not None:
        summary["expected_md5"] = expected_md5
        summary["md5"] = file_md5(path)
        if summary["md5"] != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {path}: {summary['md5']} != {expected_md5}")
    return summary


def extract_metadata(archive: Path, destination: Path, overwrite: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in tqdm(members, desc="extract FAIR metadata"):
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            with bundle.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            os.replace(temporary, target)


def locate_metadata_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def decode_matlab_strings(h5_file, references) -> list[list[str]]:
    strings = []
    for reference in references.reshape(-1):
        values = np.asarray(h5_file[reference]).reshape(-1)
        strings.append(["".join(chr(int(value)) for value in values)])
    return strings


def prepare_test_pickle(
    labeled_mat: Path,
    metadata_dir: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.is_file() and not overwrite:
        with output_path.open("rb") as handle:
            existing = pickle.load(handle)
        return summarize_test_pickle(existing, output_path)

    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "Preparing nyuv2_snorm_all.pkl requires h5py; run this script in an environment that provides it"
        ) from error

    normals_path = locate_metadata_file(metadata_dir, "all_normals.pklz")
    train_json_path = locate_metadata_file(metadata_dir, "train_SN40.json")
    test_json_path = locate_metadata_file(metadata_dir, "test_SN40.json")
    with gzip.open(normals_path, "rb") as handle:
        normal_data = pickle.load(handle)
    all_normals = np.asarray(normal_data["all_normals"])
    if all_normals.shape != (1449, 480, 640, 3):
        raise RuntimeError(f"Unexpected FAIR normals shape: {all_normals.shape}")

    with h5py.File(labeled_mat, "r") as h5_file:
        raw_depths = np.asarray(h5_file["rawDepths"])
        images = np.asarray(h5_file["images"])
        scene_types = decode_matlab_strings(h5_file, np.asarray(h5_file["sceneTypes"]))
    if raw_depths.shape != (1449, 640, 480):
        raise RuntimeError(f"Unexpected NYUv2 rawDepths shape: {raw_depths.shape}")
    if images.shape != (1449, 3, 640, 480):
        raise RuntimeError(f"Unexpected NYUv2 images shape: {images.shape}")

    train_json = json.loads(train_json_path.read_text(encoding="utf-8"))
    test_json = json.loads(test_json_path.read_text(encoding="utf-8"))
    train_indices = np.array([int(item["img"].split("_")[0]) - 1 for item in train_json])
    test_indices = np.array([int(item["img"].split("_")[0]) - 1 for item in test_json])
    payload = {
        "depths": np.transpose(raw_depths, (0, 2, 1)),
        "images": np.transpose(images, (0, 1, 3, 2)),
        "snorms": np.transpose(all_normals, (0, 3, 1, 2)),
        "scene_types": scene_types,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    summary = summarize_test_pickle(payload, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_path)
    summary["bytes"] = output_path.stat().st_size
    return summary


def summarize_test_pickle(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    expected_shapes = {
        "depths": (1449, 480, 640),
        "images": (1449, 3, 480, 640),
        "snorms": (1449, 3, 480, 640),
    }
    shapes = {name: tuple(np.asarray(payload[name]).shape) for name in expected_shapes}
    if shapes != expected_shapes:
        raise RuntimeError(f"NYUv2 test pickle shape mismatch: {shapes} != {expected_shapes}")
    train_count = len(payload["train_indices"])
    test_count = len(payload["test_indices"])
    if (train_count, test_count) != (795, 654):
        raise RuntimeError(f"NYUv2 split mismatch: train={train_count}, test={test_count}")
    overlap = np.intersect1d(payload["train_indices"], payload["test_indices"])
    if overlap.size:
        raise RuntimeError(f"NYUv2 train/test overlap contains {overlap.size} samples")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size if path.is_file() else None,
        "shapes": {name: list(shape) for name, shape in shapes.items()},
        "train_count": train_count,
        "test_count": test_count,
        "split_overlap": int(overlap.size),
    }


def extract_geonet_archive(archive: Path, destination: Path, overwrite: bool) -> int:
    expected_size = GEONET_ARCHIVE_BYTES.get(archive.name)
    if expected_size is None:
        raise RuntimeError(f"Expected GeoNet archive name data1.zip or data2.zip, got {archive.name}")
    verify_file(archive, expected_size, expected_md5=None)
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.infolist() if not member.is_dir()]
        for member in tqdm(members, desc=f"extract {archive.name}"):
            if Path(member.filename).suffix.lower() != ".mat":
                continue
            target = destination / Path(member.filename).name
            if target.name in GEONET_BAD_FILES:
                continue
            if target.is_file() and target.stat().st_size == member.file_size and not overwrite:
                continue
            temporary = target.with_name(f".{target.name}.tmp")
            with bundle.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            os.replace(temporary, target)
            written += 1
    return written


def verify_geonet(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.mat"))
    names = [path.name for path in files]
    precleaned = len(names) == GEONET_USABLE_SAMPLES
    if len(names) == GEONET_ARCHIVE_SAMPLES:
        observed_bad = tuple(names[index] for index in GEONET_BAD_SORTED_INDICES)
        if observed_bad != GEONET_BAD_FILES:
            raise RuntimeError(f"Unexpected GeoNet bad samples: {observed_bad}")
        usable = names.copy()
        for index in reversed(GEONET_BAD_SORTED_INDICES):
            del usable[index]
    elif precleaned:
        if set(names).intersection(GEONET_BAD_FILES):
            raise RuntimeError("Precleaned GeoNet root still contains bad samples")
        usable = names
    else:
        raise RuntimeError(
            f"Expected 30,916 raw or 30,914 precleaned MAT files under {root}, "
            f"got {len(names)}"
        )
    if len(usable) != GEONET_USABLE_SAMPLES:
        raise RuntimeError(f"Expected 30,914 usable GeoNet samples, got {len(usable)}")
    return {
        "root": str(root.resolve()),
        "archive_count": GEONET_ARCHIVE_SAMPLES,
        "on_disk_count": len(files),
        "usable_count": len(usable),
        "precleaned": precleaned,
        "removed_sorted_indices": list(GEONET_BAD_SORTED_INDICES),
        "removed_files": list(GEONET_BAD_FILES),
    }


def main() -> None:
    args = parse_args()
    root = args.nyu_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    labeled_mat = (args.labeled_mat or root / "nyu_depth_v2_labeled.mat").resolve()
    normals_archive = (
        args.normals_archive or root / "nyuv2_surfacenormal_metadata.zip"
    ).resolve()
    metadata_dir = (args.metadata_dir or root / "surfacenormal_metadata").resolve()
    test_pickle = (args.test_pickle or root / "nyuv2_snorm_all.pkl").resolve()
    geonet_root = (args.geonet_root or root / "nyuv2_geonet").resolve()
    manifest_path = (args.manifest or root / "probe3d_nyuv2_manifest.json").resolve()

    summary: dict[str, Any] = {
        "labeled_mat": verify_file(labeled_mat, LABELED_MAT_BYTES, expected_md5=None),
        "normals_archive": verify_file(
            normals_archive,
            NORMALS_ARCHIVE_BYTES,
            expected_md5=None if args.skip_hashes else NORMALS_ARCHIVE_MD5,
        ),
    }
    if not args.verify_only and not args.skip_test_pickle:
        extract_metadata(normals_archive, metadata_dir, args.overwrite)
        summary["test_pickle"] = prepare_test_pickle(
            labeled_mat,
            metadata_dir,
            test_pickle,
            args.overwrite,
        )
    elif test_pickle.is_file():
        with test_pickle.open("rb") as handle:
            summary["test_pickle"] = summarize_test_pickle(pickle.load(handle), test_pickle)

    if not args.skip_geonet:
        if not args.verify_only:
            summary["geonet_archives"] = []
            for archive in args.geonet_archive:
                written = extract_geonet_archive(archive.resolve(), geonet_root, args.overwrite)
                summary["geonet_archives"].append(
                    {"path": str(archive.resolve()), "files_written": written}
                )
        summary["geonet"] = verify_geonet(geonet_root)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
