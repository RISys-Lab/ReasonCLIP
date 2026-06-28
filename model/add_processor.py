import argparse
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, upload_folder


def _read_bash_array(path: Path, name: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"{name}=\(\s*(.*?)\s*\)", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array: {name}")
    return re.findall(r'"([^"]+)"', match.group(1))


def infer_target_repo(source_repo: str, namespace: str) -> str:
    name = source_repo.split("/", 1)[1]

    if name.startswith("clip-r-336-"):
        prefix = "RC-L14-336"
    elif name.startswith("clip-r-b32-"):
        prefix = "RC-B32"
    elif name.startswith("clip-r-"):
        prefix = "RC-L14-224"
    elif name.startswith("siglip2-r-go-"):
        prefix = "RS2-GO16"
    elif name.startswith("siglip2-r-"):
        prefix = "RS2-So14"
    elif name.startswith("siglip-r-"):
        prefix = "RS-So14"
    else:
        raise ValueError(f"Cannot infer target repo for source: {source_repo}")

    if "-s1-read" in name:
        suffix = "READ"
    elif "-rea-" in name:
        suffix = "S0-Rea"
    elif "-des-" in name:
        suffix = "S0-Des"
    elif "-s1-" in name:
        suffix = "S1"
    elif "-s2-" in name:
        suffix = "S2"
    else:
        raise ValueError(f"Cannot infer training stage for source: {source_repo}")

    return f"{namespace}/{prefix}-{suffix}"


def add_processor(
    source_repo: str,
    processor_repo: str,
    target_repo: str,
    token: str | None,
) -> None:
    from transformers import AutoProcessor

    api = HfApi(token=token)
    api.create_repo(target_repo, repo_type="model", private=False, exist_ok=True)
    if hasattr(api, "update_repo_visibility"):
        api.update_repo_visibility(repo_id=target_repo, repo_type="model", private=False)

    with tempfile.TemporaryDirectory(prefix="reasonclip_hf_") as tmpdir:
        local_dir = Path(tmpdir) / "model"
        snapshot_download(
            repo_id=source_repo,
            repo_type="model",
            local_dir=local_dir,
            token=token,
        )
        upload_folder(
            repo_id=target_repo,
            repo_type="model",
            folder_path=local_dir,
            token=token,
            commit_message=f"Copy weights from {source_repo}",
        )

    processor = AutoProcessor.from_pretrained(processor_repo, trust_remote_code=True, token=token)
    processor.push_to_hub(target_repo, token=token, private=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add processors to released HF repos.")
    parser.add_argument("--model-list", type=Path, default=Path("model/models_pre_release.sh"))
    parser.add_argument("--namespace", default="RISys-Lab", help="Target Hugging Face namespace.")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print source/processor/target mapping only.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based index to start from.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive zero-based index to stop at.")
    args = parser.parse_args()

    sources = _read_bash_array(args.model_list, "models")
    processors = _read_bash_array(args.model_list, "processors")
    if len(sources) != len(processors):
        raise ValueError(f"models/processors length mismatch: {len(sources)} != {len(processors)}")

    end_index = args.end_index if args.end_index is not None else len(sources)
    for source_repo, processor_repo in zip(sources[args.start_index:end_index], processors[args.start_index:end_index]):
        target_repo = infer_target_repo(source_repo, args.namespace)
        print(f"{source_repo} -> {target_repo} | processor={processor_repo}")
        if not args.dry_run:
            add_processor(
                source_repo=source_repo,
                processor_repo=processor_repo,
                target_repo=target_repo,
                token=args.token,
            )


if __name__ == "__main__":
    main()
