# Reasoning Geometry Analysis

This directory contains the DOCCI/Visual Genome analysis pipeline for checking whether ReasonCLIP image embeddings preserve base visual alignment while better matching visually grounded commonsense reasoning structure.

Use the requested environment:

```bash
/home/localadmin/venvs/llm/bin/python
```

The pipeline is versioned:

- `v0`: smoke run, usually 100 images.
- `v1`: iteration run, usually 1K-2K images.
- `v2`: formal run, target 5K images.

Generated data lives under `reasoning_geometry/work/`, `reasoning_geometry/artifacts/`, and `reasoning_geometry/explorer/data/`; these paths are ignored by git.

## Minimal Run

```bash
PY=/home/localadmin/venvs/llm/bin/python

$PY reasoning_geometry/scripts/prepare_docci.py \
  --input /home/localadmin/bz/RCLIP/docci_pairs_100.jsonl \
  --output reasoning_geometry/work/v0/metadata.jsonl \
  --limit 100

$PY reasoning_geometry/scripts/annotate_reasoning_captions.py \
  --input reasoning_geometry/work/v0/metadata.jsonl \
  --output reasoning_geometry/work/v0/annotations.jsonl

$PY reasoning_geometry/scripts/embed_captions.py \
  --input reasoning_geometry/work/v0/annotations.jsonl \
  --output-dir reasoning_geometry/work/v0/caption_embeddings \
  --method hf

$PY reasoning_geometry/scripts/embed_images.py \
  --input reasoning_geometry/work/v0/annotations.jsonl \
  --output-dir reasoning_geometry/work/v0/image_embeddings \
  --models clip_base,clip_s1,clip_s2,siglip_base,siglip_s1,siglip_s2

$PY reasoning_geometry/scripts/compute_metrics.py \
  --annotations reasoning_geometry/work/v0/annotations.jsonl \
  --caption-embeddings reasoning_geometry/work/v0/caption_embeddings/embeddings.npy \
  --image-embedding-dir reasoning_geometry/work/v0/image_embeddings \
  --output reasoning_geometry/work/v0/metrics.json

$PY reasoning_geometry/scripts/compute_retrievals.py \
  --annotations reasoning_geometry/work/v0/annotations.jsonl \
  --image-embedding-dir reasoning_geometry/work/v0/image_embeddings \
  --output reasoning_geometry/work/v0/retrievals.json

$PY reasoning_geometry/scripts/build_explorer.py \
  --annotations reasoning_geometry/work/v0/annotations.jsonl \
  --caption-embeddings reasoning_geometry/work/v0/caption_embeddings/embeddings.npy \
  --image-embedding-dir reasoning_geometry/work/v0/image_embeddings \
  --metrics reasoning_geometry/work/v0/metrics.json \
  --retrievals reasoning_geometry/work/v0/retrievals.json \
  --output-dir reasoning_geometry/explorer
```

Then open `reasoning_geometry/explorer/index.html`.

## Current Formal Run

The completed formal run is `v2` with 5,000 DOCCI images.

Key files:

- `reasoning_geometry/report_v2.md`: Chinese analysis report.
- `reasoning_geometry/work/v2/metrics.json`: final metrics.
- `reasoning_geometry/work/v2/retrievals.json`: prompt retrieval results.
- `reasoning_geometry/explorer/index.html`: latest generated static explorer.

Caption reference for V1/V2 is `Qwen/Qwen3-1.7B` mean-pooled hidden states, used only as an independent language-only semantic space.
