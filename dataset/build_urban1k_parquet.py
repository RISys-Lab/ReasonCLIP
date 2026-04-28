import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Image, Value


def parse_args():
    parser = argparse.ArgumentParser(description="Build Urban1k parquet with embedded image bytes.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/Urban1k/raw"),
        help="Directory containing data.json and image/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Urban1k/data/test-00000-of-00001.parquet"),
        help="Output parquet path.",
    )
    return parser.parse_args()


def load_records(raw_dir: Path):
    json_path = raw_dir / "data.json"
    image_dir = raw_dir / "image"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = []
    for item in metadata:
        image_name = item.get("image_name") or item.get("filename")
        caption = item.get("caption")
        if not image_name:
            raise ValueError(f"Missing image_name/filename in item: {item}")
        if caption is None:
            raise ValueError(f"Missing caption for image: {image_name}")

        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        records.append(
            {
                "id": int(item["id"]) if "id" in item else len(records) + 1,
                "image_name": image_name,
                "caption": caption,
                "image": {"bytes": image_path.read_bytes(), "path": image_name},
            }
        )

    return records


def main():
    args = parse_args()
    records = load_records(args.raw_dir)
    features = Features(
        {
            "id": Value("int32"),
            "image_name": Value("string"),
            "caption": Value("string"),
            "image": Image(),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(records, features=features)
    dataset.to_parquet(str(args.output))
    print(f"Wrote {len(dataset)} examples to {args.output}")


if __name__ == "__main__":
    main()
