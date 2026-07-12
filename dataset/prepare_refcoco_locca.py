#!/usr/bin/env python3
"""Prepare the full and clean RefCOCO mixtures used by LocCa and SigLIP 2.

The split construction follows the official Big Vision RefCOCO builder: merge
RefCOCO-UNC, RefCOCO+-UNC, and RefCOCOg-UMD. ``train_full`` preserves the
standard cross-dataset image overlap used by the paper table. ``train_clean``
removes every training image that appears in any validation or test split of
any of the three datasets. Each referring sentence is one training example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFER_ROOT = REPO_ROOT / "rebuttal" / "downstream_repos" / "refer" / "data"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "rebuttal" / "downstream_data" / "COCO" / "train2014"
DEFAULT_OUT_ROOT = REPO_ROOT / "rebuttal" / "downstream_data" / "RefCOCOLocCa"

DATASETS = {
    "refcoco": {"directory": "refcoco", "split_by": "unc", "eval_splits": ("val", "testA", "testB")},
    "refcocoplus": {
        "directory": "refcoco+",
        "split_by": "unc",
        "eval_splits": ("val", "testA", "testB"),
    },
    "refcocog": {"directory": "refcocog", "split_by": "umd", "eval_splits": ("val", "test")},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refer-root", type=Path, default=DEFAULT_REFER_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe path in {archive}: {member.filename}")
        handle.extractall(destination)


def ensure_annotations(refer_root: Path) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}
    for dataset, config in DATASETS.items():
        directory = refer_root / config["directory"]
        archive = refer_root / f"{config['directory']}.zip"
        refs_path = directory / f"refs({config['split_by']}).p"
        instances_path = directory / "instances.json"
        if not refs_path.is_file() or not instances_path.is_file():
            if not archive.is_file():
                raise FileNotFoundError(
                    f"Missing {refs_path} or {instances_path}, and no archive at {archive}"
                )
            safe_extract(archive, refer_root)
        if not refs_path.is_file() or not instances_path.is_file():
            raise FileNotFoundError(f"Incomplete extracted annotations under {directory}")
        sources[dataset] = {
            "directory": directory,
            "archive": archive,
            "refs": refs_path,
            "instances": instances_path,
        }
    return sources


def load_pickle(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        records = pickle.load(handle, encoding="latin1")
    if not isinstance(records, list):
        raise TypeError(f"Expected list in {path}, got {type(records).__name__}")
    return records


def load_instances(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations = {int(item["id"]): item for item in payload["annotations"]}
    if len(images) != len(payload["images"]) or len(annotations) != len(payload["annotations"]):
        raise RuntimeError(f"Duplicate image or annotation IDs in {path}")
    return images, annotations


def validate_box(box: Iterable[float], width: int, height: int, context: str) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError(f"Expected xywh box in {context}, got {values}")
    x, y, box_w, box_h = values
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Non-positive box in {context}: {values}")
    tolerance = 1.0
    if x < -tolerance or y < -tolerance or x + box_w > width + tolerance or y + box_h > height + tolerance:
        raise ValueError(f"Out-of-bounds box in {context}: {values} for {width}x{height}")
    return values


def expand_ref(
    dataset: str,
    ref: dict[str, Any],
    images: dict[int, dict[str, Any]],
    annotations: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    image_id = int(ref["image_id"])
    ann_id = int(ref["ann_id"])
    image = images.get(image_id)
    annotation = annotations.get(ann_id)
    if image is None or annotation is None:
        raise KeyError(f"Missing image/annotation for {dataset} ref {ref.get('ref_id')}")
    if int(annotation["image_id"]) != image_id:
        raise RuntimeError(f"Annotation {ann_id} points to the wrong image")
    width = int(image["width"])
    height = int(image["height"])
    box = validate_box(annotation["bbox"], width, height, f"{dataset}/{ref.get('ref_id')}")
    output = []
    for sentence in ref["sentences"]:
        text = str(sentence.get("sent", sentence.get("raw", ""))).strip()
        if not text:
            raise ValueError(f"Empty sentence in {dataset} ref {ref.get('ref_id')}")
        output.append(
            {
                "dataset": dataset,
                "split": str(ref["split"]),
                "split_by": DATASETS[dataset]["split_by"],
                "ref_id": int(ref["ref_id"]),
                "sentence_id": int(sentence["sent_id"]),
                "ann_id": ann_id,
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "width": width,
                "height": height,
                "bbox_xywh": box,
                "expression": text,
            }
        )
    return output


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def source_summary(paths: dict[str, Path]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "refs": str(paths["refs"].resolve()),
        "refs_bytes": paths["refs"].stat().st_size,
        "instances": str(paths["instances"].resolve()),
        "instances_bytes": paths["instances"].stat().st_size,
    }
    archive = paths["archive"]
    if archive.is_file():
        output.update(
            archive=str(archive.resolve()),
            archive_bytes=archive.stat().st_size,
            archive_md5=md5sum(archive),
        )
    return output


def main() -> None:
    args = parse_args()
    sources = ensure_annotations(args.refer_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    refs_by_dataset: dict[str, list[dict[str, Any]]] = {}
    images_by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    annotations_by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset, paths in sources.items():
        refs_by_dataset[dataset] = load_pickle(paths["refs"])
        images, annotations = load_instances(paths["instances"])
        images_by_dataset[dataset] = images
        annotations_by_dataset[dataset] = annotations

    all_refs = [ref for refs in refs_by_dataset.values() for ref in refs]
    excluded_image_ids = {int(ref["image_id"]) for ref in all_refs if str(ref["split"]) != "train"}
    clean_refs = {
        dataset: [
            ref
            for ref in refs
            if str(ref["split"]) == "train" and int(ref["image_id"]) not in excluded_image_ids
        ]
        for dataset, refs in refs_by_dataset.items()
    }
    leaked = {
        int(ref["image_id"])
        for refs in clean_refs.values()
        for ref in refs
        if int(ref["image_id"]) in excluded_image_ids
    }
    if leaked:
        raise RuntimeError(f"Clean RefCOCO train split leaked {len(leaked)} held-out images")

    full_train_records = []
    for dataset in ("refcocog", "refcoco", "refcocoplus"):
        for ref in refs_by_dataset[dataset]:
            if str(ref["split"]) != "train":
                continue
            full_train_records.extend(
                expand_ref(
                    dataset,
                    ref,
                    images_by_dataset[dataset],
                    annotations_by_dataset[dataset],
                )
            )
    train_records = [
        record
        for record in full_train_records
        if int(record["image_id"]) not in excluded_image_ids
    ]
    full_train_count = write_jsonl(args.out_root / "train_full.jsonl", full_train_records)
    train_count = write_jsonl(args.out_root / "train_clean.jsonl", train_records)

    eval_counts: dict[str, int] = {}
    eval_image_ids: set[int] = set()
    eval_records: list[dict[str, Any]] = []
    for dataset, config in DATASETS.items():
        for split in config["eval_splits"]:
            split_records = []
            for ref in refs_by_dataset[dataset]:
                if str(ref["split"]) == split:
                    split_records.extend(
                        expand_ref(
                            dataset,
                            ref,
                            images_by_dataset[dataset],
                            annotations_by_dataset[dataset],
                        )
                    )
            if not split_records:
                raise RuntimeError(f"No examples for {dataset}/{split}")
            name = f"{dataset}_{split}"
            eval_counts[name] = write_jsonl(args.out_root / f"{name}.jsonl", split_records)
            eval_records.extend(split_records)
            eval_image_ids.update(int(record["image_id"]) for record in split_records)
    eval_count = write_jsonl(args.out_root / "eval_all.jsonl", eval_records)

    required_image_names = {
        record["file_name"] for record in full_train_records
    } | {
        images_by_dataset[dataset][image_id]["file_name"]
        for dataset in DATASETS
        for image_id in eval_image_ids
        if image_id in images_by_dataset[dataset]
    }
    missing_images = sorted(name for name in required_image_names if not (args.image_root / name).is_file())
    if missing_images and not args.skip_image_check:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} COCO images under {args.image_root}; first: {preview}"
        )

    source_counts = {}
    full_counts = {}
    clean_counts = {}
    for dataset, refs in refs_by_dataset.items():
        split_refs = Counter(str(ref["split"]) for ref in refs)
        split_sentences = Counter()
        split_images: dict[str, set[int]] = {}
        for ref in refs:
            split = str(ref["split"])
            split_sentences[split] += len(ref["sentences"])
            split_images.setdefault(split, set()).add(int(ref["image_id"]))
        source_counts[dataset] = {
            "refs": dict(sorted(split_refs.items())),
            "sentences": dict(sorted(split_sentences.items())),
            "images": {key: len(value) for key, value in sorted(split_images.items())},
        }
        full_counts[dataset] = {
            "refs": split_refs["train"],
            "sentences": split_sentences["train"],
            "images": len(split_images["train"]),
        }
        clean_counts[dataset] = {
            "refs": len(clean_refs[dataset]),
            "sentences": sum(len(ref["sentences"]) for ref in clean_refs[dataset]),
            "images": len({int(ref["image_id"]) for ref in clean_refs[dataset]}),
        }

    full_train_image_ids = {int(record["image_id"]) for record in full_train_records}
    clean_train_image_ids = {int(record["image_id"]) for record in train_records}
    full_eval_overlap = full_train_image_ids.intersection(eval_image_ids)
    full_eval_overlap_by_split = {}
    for name in eval_counts:
        split_records = [
            record
            for record in eval_records
            if "{}_{}".format(record["dataset"], record["split"]) == name
        ]
        split_image_ids = {int(record["image_id"]) for record in split_records}
        full_eval_overlap_by_split[name] = {
            "images": len(split_image_ids),
            "overlap_images": len(split_image_ids.intersection(full_train_image_ids)),
            "expressions": len(split_records),
            "overlap_expressions": sum(
                int(record["image_id"]) in full_train_image_ids
                for record in split_records
            ),
        }
    if clean_train_image_ids.intersection(eval_image_ids):
        raise RuntimeError("Clean RefCOCO records overlap held-out evaluation images")

    manifest = {
        "protocol": {
            "name": "LocCa/SigLIP2 frozen-encoder RefCOCO REC",
            "datasets": ["refcocog_umd", "refcoco_unc", "refcocoplus_unc"],
            "full_rule": "merge all dataset train splits without cross-dataset de-duplication",
            "clean_rule": "exclude from train every image occurring in any non-train split",
            "training_unit": "one referring sentence",
            "split_overlap": len(leaked),
        },
        "sources": {dataset: source_summary(paths) for dataset, paths in sources.items()},
        "source_counts": source_counts,
        "full_counts": full_counts,
        "excluded_images": len(excluded_image_ids),
        "clean_counts": clean_counts,
        "full_train_sentences": full_train_count,
        "full_train_images": len(full_train_image_ids),
        "full_train_eval_image_overlap": len(full_eval_overlap),
        "full_train_eval_overlap_by_split": full_eval_overlap_by_split,
        "train_sentences": train_count,
        "train_images": len(clean_train_image_ids),
        "eval_sentences": eval_counts,
        "eval_sentences_total": eval_count,
        "eval_images": len(eval_image_ids),
        "image_root": str(args.image_root.resolve()),
        "required_images": len(required_image_names),
        "missing_images": len(missing_images),
        "image_check_skipped": bool(args.skip_image_check),
    }
    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
