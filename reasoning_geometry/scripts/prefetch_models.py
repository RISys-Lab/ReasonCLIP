#!/usr/bin/env python3
import argparse
import os
import time
from typing import Iterable, List

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import snapshot_download

from common import IMAGE_MODELS, parse_model_keys

ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "*.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "added_tokens.json",
]


def unique_repos(model_keys: Iterable[str], include_processors: bool = True) -> List[str]:
    repos: List[str] = []
    for key in model_keys:
        cfg = IMAGE_MODELS[key]
        candidates = [cfg["model_id"]]
        if include_processors:
            candidates.insert(0, cfg["processor_id"])
        for repo in candidates:
            if repo not in repos:
                repos.append(repo)
    return repos


def download_repo(repo: str, retries: int, sleep_seconds: int, local_files_only: bool) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[{attempt}/{retries}] downloading {repo}", flush=True)
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=ALLOW_PATTERNS,
                local_files_only=local_files_only,
                max_workers=1,
            )
            print(f"done {repo}: {path}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"failed {repo} attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to download {repo} after {retries} attempts") from last_error


def main():
    parser = argparse.ArgumentParser(description="Prefetch HF model/processor files for reasoning geometry runs.")
    parser.add_argument("--models", default="all")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-processors", action="store_true", help="Only download model repos, not processor/base repos.")
    args = parser.parse_args()
    repos = unique_repos(parse_model_keys(args.models), include_processors=not args.skip_processors)
    for repo in repos:
        download_repo(repo, args.retries, args.sleep_seconds, args.local_files_only)
    print("all requested repos are available")


if __name__ == "__main__":
    main()
