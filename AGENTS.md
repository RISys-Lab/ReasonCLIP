# Repository Guidelines

## Project Structure & Module Organization
Core code is organized by workflow:
- `trainning/`: model fine-tuning entrypoints (CLIP/SigLIP variants, stage-specific scripts like `ft_clip_r_s1.py`).
- `eval/`: evaluation pipelines (`eval_retrieval.py`, `eval_zeroshot_imagenet.py`, `eval_winogavil.py`) plus static eval resources in `eval/eval_data/`.
- `dataset/`: data generation/reformat/cleaning utilities and task configs.
- `scripts/`: runnable shell jobs (mostly SLURM + `accelerate launch`) for training and evaluation.
- `utils/`: shared helpers and prompt utilities.
- `model/`: model reference lists (for example `hf_models.txt`).

## Build, Test, and Development Commands
No packaged build system is defined; use script and Python entrypoints directly.
- `bash scripts/ft_siglip_unifire.sh`: launch a standard multi-GPU fine-tuning run.
- `bash scripts/eval_imagenet.sh`: run zero-shot ImageNet-family evaluations.
- `bash scripts/eval_retrieval.sh`: run retrieval benchmarks (Urban1k/COCO/Flickr variants).
- `python eval/eval_cli.py`: minimal inference smoke check against a hosted model.
- `python dataset/gen.py --help`: inspect dataset generation arguments.

If running outside SLURM, adapt paths and environment variables (`HF_HOME`, dataset/model paths, CUDA device settings).

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation and `snake_case` for functions/variables. Keep CLI scripts `argparse`-driven and favor explicit flags over hardcoded constants. Name new training/eval scripts consistently with existing patterns (`ft_*`, `eval_*`). For shell scripts, keep `set -euo pipefail` and uppercase env vars.

## Testing Guidelines
There is currently no dedicated `tests/` suite. Treat evaluation and CLI runs as regression checks:
- Run at least one small eval (`python eval/eval_zeroshot_imagenet.py --help` plus a short real run).
- For training changes, validate argument parsing and one short sanity run on a subset.
- Record output directories and key metrics in PR notes for reproducibility.

## Commit & Pull Request Guidelines
Recent history uses short subjects (`upd`, `upd: ...`, `fix: ...`). Keep commits concise, imperative, and scoped (for example: `fix: handle missing local images in retrieval eval`). PRs should include:
- what changed and why,
- exact commands used for validation,
- dataset/model paths assumptions,
- before/after metrics or logs for training/eval changes.
