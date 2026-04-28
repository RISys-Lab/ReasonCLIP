import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image, List, Value


DEFAULT_SPLITS = {
    "v1": "rclip_5k_v1_gpt_new.jsonl",
    "v2": "rclip_5k_v2_gpt5_new_v2.jsonl",
    "v3": "rclip_5k_v3_gpt_new.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build RCLIP parquet splits with embedded image bytes.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/docci_images/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/RCLIP/data"))
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def normalize_sets(sets):
    normalized = []
    for item in sets:
        tag = item.get("tag", "")
        gt = item.get("gt", "")
        neg = item.get("neg", [])
        if not isinstance(tag, str) or not tag:
            raise ValueError("RCLIP set is missing tag")
        if not isinstance(gt, str) or not gt.strip():
            raise ValueError("RCLIP set is missing gt")
        if not isinstance(neg, list) or len(neg) != 4:
            raise ValueError("RCLIP set neg must contain four strings")
        normalized.append(
            {
                "tag": tag,
                "gt": gt.strip(),
                "neg": [str(x).strip() for x in neg],
            }
        )
    return normalized


def build_records(jsonl_path: Path, image_dir: Path):
    records = []
    for sample in read_jsonl(jsonl_path):
        image_name = Path(sample["image_path"]).name
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for {sample.get('id')}: {image_path}")

        records.append(
            {
                "id": str(sample.get("id", "")),
                "image_name": image_name,
                "text": str(sample.get("text", "")),
                "version": str(sample.get("version", "")),
                "sets": normalize_sets(sample.get("sets", [])),
                "image": {"bytes": image_path.read_bytes(), "path": image_name},
            }
        )
    return records


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = Features(
        {
            "id": Value("string"),
            "image_name": Value("string"),
            "text": Value("string"),
            "version": Value("string"),
            "sets": List(
                {
                    "tag": Value("string"),
                    "gt": Value("string"),
                    "neg": List(Value("string")),
                }
            ),
            "image": Image(),
        }
    )

    for split, filename in DEFAULT_SPLITS.items():
        records = build_records(args.data_dir / filename, args.image_dir)
        dataset = Dataset.from_list(records, features=features)
        output_path = args.output_dir / f"{split}-00000-of-00001.parquet"
        dataset.to_parquet(str(output_path))
        print(f"Wrote {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
