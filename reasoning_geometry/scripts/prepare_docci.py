#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import first_sentence, read_jsonl, seeded_rows, write_jsonl


def build_row(row):
    image_id = row.get("id") or row.get("image_id") or row.get("example_id")
    if not image_id:
        raise ValueError(f"Missing image id in row: {row}")
    source_caption = row.get("text") or row.get("caption") or row.get("source_caption") or ""
    out = {
        "image_id": image_id,
        "split": row.get("split", "unknown"),
        "image_path": row.get("image_path") or row.get("image") or row.get("path"),
        "image_file": row.get("image_file") or Path(row.get("image_path", "")).name,
        "source_caption": source_caption,
        "descriptive_caption": first_sentence(source_caption),
        "source_dataset": "docci",
        "metadata": row.get("metadata", {}),
    }
    if row.get("subset_tag"):
        out["subset_tag"] = row["subset_tag"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare DOCCI records for reasoning geometry analysis.")
    parser.add_argument("--input", required=True, help="Input DOCCI jsonl, e.g. docci_pairs_5k.jsonl")
    parser.add_argument("--output", required=True, help="Output metadata jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to keep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    rows = seeded_rows(read_jsonl(args.input), args.limit, args.seed, args.shuffle)
    prepared = []
    missing = 0
    for row in rows:
        out = build_row(row)
        image_path = out.get("image_path")
        if not image_path or not Path(image_path).exists():
            missing += 1
            if not args.allow_missing_images:
                raise FileNotFoundError(f"Missing image for {out['image_id']}: {image_path}")
        prepared.append(out)

    write_jsonl(args.output, prepared)
    print(f"wrote {len(prepared)} rows to {args.output}; missing_images={missing}")


if __name__ == "__main__":
    main()
