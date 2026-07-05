# Data Schema

## Prepared Metadata

Each line in `metadata.jsonl`:

```json
{
  "image_id": "train_05276",
  "split": "train",
  "image_path": "/abs/path/to/image.jpg",
  "image_file": "train_05276.jpg",
  "source_caption": "Long DOCCI description...",
  "descriptive_caption": "A shorter descriptive caption.",
  "source_dataset": "docci",
  "metadata": {}
}
```

## Reasoning Annotations

Each line in `annotations.jsonl`:

```json
{
  "image_id": "train_05276",
  "image_path": "/abs/path/to/image.jpg",
  "source_caption": "Long DOCCI description...",
  "descriptive_caption": "A shorter descriptive caption.",
  "reasoning_captions": [
    {
      "caption": "The car appears parked and unattended because it is stopped at the curb with static details visible.",
      "visible_evidence": "the car is parked on a street curb",
      "commonsense_conclusion": "the vehicle is not currently being driven",
      "reasoning_type": "state_or_activity"
    }
  ],
  "annotation_method": "docci_text_rules_v1",
  "annotation_model": "codex_text_heuristic",
  "annotation_prompt_version": "reasoning_caption_v1"
}
```

These are LLM/rule generated reasoning annotations used as an independent semantic reference, not human ground truth.

## Embeddings

Embedding directories contain:

- `ids.json`: image ids in row order.
- `embeddings.npy`: normalized embedding matrix.
- `{model_key}.npy`: normalized image embedding matrix for one visual encoder.
- `{model_key}.meta.json`: model provenance and dimensions.
