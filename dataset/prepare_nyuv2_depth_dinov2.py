#!/usr/bin/env python3
"""Extract and verify the BTS NYUv2 depth subset used by DINOv2/TIPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from tqdm import tqdm


ARCHIVE_BYTES = 6_683_052_232
ARCHIVE_ENTRIES = 75_873
ARCHIVE_FILES = 75_554
ARCHIVE_RGB_FILES = 37_776
ARCHIVE_DEPTH_FILES = 37_776
SPLIT_COUNTS = {"train": 24_231, "test": 654}
SPLIT_SHA256 = {
    "train": "11dcd508d3c669b2fed27eff2919a076484eea8ab6b1f889f32da6fc0a395354",
    "test": "e92c191d5835a22589b843c11b80cd7235d5bd0d4136e42a6debcea4511bf906",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("rebuttal/downstream_data/NYUv2/dinov2_nyu")
    parser.add_argument("--archive", type=Path, default=base / "nyu.zip")
    parser.add_argument("--output-root", type=Path, default=base / "NYU")
    parser.add_argument("--manifest", type=Path, default=base / "manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-archive-sha256", action="store_true")
    parser.add_argument("--skip-archive-crc", action="store_true")
    parser.add_argument("--skip-dimensions", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def split_records(data: bytes, split: str) -> list[tuple[str, str, float]]:
    digest = hashlib.sha256(data).hexdigest()
    if digest != SPLIT_SHA256[split]:
        raise RuntimeError(f"Unexpected nyu_{split}.txt SHA-256: {digest}")
    records = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError(f"Malformed nyu_{split}.txt:{line_number}")
        image_text, depth_text, focal_text = fields
        image_path = PurePosixPath(image_text.lstrip("/"))
        depth_path = PurePosixPath(depth_text.lstrip("/"))
        if ".." in image_path.parts or ".." in depth_path.parts:
            raise RuntimeError(f"Unsafe split path at nyu_{split}.txt:{line_number}")
        records.append((image_path.as_posix(), depth_path.as_posix(), float(focal_text)))
    expected = SPLIT_COUNTS[split]
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected:,} {split} records, got {len(records):,}")
    if len({record[0] for record in records}) != len(records):
        raise RuntimeError(f"Duplicate RGB paths in nyu_{split}.txt")
    return records


def extract_member(
    bundle: zipfile.ZipFile,
    member_name: str,
    target: Path,
    overwrite: bool,
) -> bool:
    member = bundle.getinfo(member_name)
    if target.is_file() and target.stat().st_size == member.file_size and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with bundle.open(member) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    os.replace(temporary, target)
    return True


def verify_dimensions(
    root: Path,
    records: list[tuple[str, str, float]],
) -> None:
    for image_name, depth_name, _ in tqdm(records, desc="verify NYUv2 dimensions"):
        with Image.open(root / image_name) as image:
            image_size = image.size
        with Image.open(root / depth_name) as depth:
            depth_size = depth.size
        if image_size != (640, 480) or depth_size != (640, 480):
            raise RuntimeError(
                f"Unexpected NYUv2 dimensions for {image_name}: {image_size}, {depth_size}"
            )


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(
            f"Archive size mismatch: {archive.stat().st_size} != {ARCHIVE_BYTES}"
        )

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        files = [member for member in infos if not member.is_dir()]
        if len(infos) != ARCHIVE_ENTRIES or len(files) != ARCHIVE_FILES:
            raise RuntimeError(
                f"Unexpected archive members: entries={len(infos)}, files={len(files)}"
            )
        rgb_count = sum(member.filename.lower().endswith(".jpg") for member in files)
        depth_count = sum(member.filename.lower().endswith(".png") for member in files)
        if (rgb_count, depth_count) != (ARCHIVE_RGB_FILES, ARCHIVE_DEPTH_FILES):
            raise RuntimeError(f"Unexpected archive RGB/depth counts: {rgb_count}, {depth_count}")
        if not args.skip_archive_crc:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Archive CRC failure: {bad_member}")

        split_bytes = {
            split: bundle.read(f"nyu/nyu_{split}.txt") for split in SPLIT_COUNTS
        }
        splits = {
            split: split_records(data, split) for split, data in split_bytes.items()
        }
        train_images = {record[0] for record in splits["train"]}
        test_images = {record[0] for record in splits["test"]}
        overlap = train_images.intersection(test_images)
        if overlap:
            raise RuntimeError(f"NYUv2 train/test RGB overlap: {len(overlap)}")

        records = splits["train"] + splits["test"]
        required_paths = {path for record in records for path in record[:2]}
        archive_names = {member.filename for member in files}
        missing_members = sorted(
            relative for relative in required_paths if f"nyu/{relative}" not in archive_names
        )
        if missing_members:
            raise RuntimeError(
                f"Archive lacks {len(missing_members)} split members; first={missing_members[0]}"
            )

        written = 0
        if not args.verify_only:
            output_root.mkdir(parents=True, exist_ok=True)
            for split, data in split_bytes.items():
                target = output_root / f"nyu_{split}.txt"
                member_name = f"nyu/nyu_{split}.txt"
                written += extract_member(bundle, member_name, target, args.overwrite)
            for relative in tqdm(sorted(required_paths), desc="extract NYUv2 depth"):
                written += extract_member(
                    bundle,
                    f"nyu/{relative}",
                    output_root / relative,
                    args.overwrite,
                )

        for split in SPLIT_COUNTS:
            output_split = output_root / f"nyu_{split}.txt"
            if not output_split.is_file():
                raise FileNotFoundError(output_split)
            split_records(output_split.read_bytes(), split)
        missing_output = sorted(
            relative for relative in required_paths if not (output_root / relative).is_file()
        )
        if missing_output:
            raise RuntimeError(
                f"Output lacks {len(missing_output)} split files; first={missing_output[0]}"
            )
        wrong_sizes = [
            relative
            for relative in required_paths
            if (output_root / relative).stat().st_size
            != bundle.getinfo(f"nyu/{relative}").file_size
        ]
        if wrong_sizes:
            raise RuntimeError(
                f"Output has {len(wrong_sizes)} size mismatches; first={wrong_sizes[0]}"
            )

    if not args.skip_dimensions:
        verify_dimensions(output_root, records)
    summary: dict[str, Any] = {
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": None if args.skip_archive_sha256 else file_sha256(archive),
            "entries": ARCHIVE_ENTRIES,
            "files": ARCHIVE_FILES,
            "rgb_files": ARCHIVE_RGB_FILES,
            "depth_files": ARCHIVE_DEPTH_FILES,
            "crc_checked": not args.skip_archive_crc,
        },
        "output_root": str(output_root),
        "files_written": written,
        "extracted_pairs": len(records),
        "train_test_overlap": len(overlap),
        "dimensions_checked": not args.skip_dimensions,
        "splits": {
            split: {
                "records": len(splits[split]),
                "sha256": SPLIT_SHA256[split],
            }
            for split in SPLIT_COUNTS
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.verify_only or not manifest_path.is_file():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(f"Wrote {manifest_path}")
    else:
        print(f"Verified without replacing {manifest_path}")


if __name__ == "__main__":
    main()
