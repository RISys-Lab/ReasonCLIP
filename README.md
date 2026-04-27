<div align="center">
  <img src="asset/reasonclip_wordmark_compact.svg" alt="ReasonCLIP" width="100%">
</div>

<div align="center">

📄 **[Paper](https://img.shields.io/badge/Paper-TODO-gray)** |  
🤗 **[Models](https://img.shields.io/badge/Models-TODO-gray)** |  
🤗 **[Dataset](https://img.shields.io/badge/Dataset-TODO-gray)** |  
🤗 **[BenchMark](https://img.shields.io/badge/Dataset-TODO-gray)** |  
📊 **[Model Card](https://img.shields.io/badge/Models-TODO-gray)**
</div>
## News

- `[TODO date]` Release ReasonCLIP Datasets, Benchmark and Models.


<!-- <details>
<summary>More</summary>

- `[TODO older date]` Older update 1.
- `[TODO older date]` Older update 2.

</details> -->

---
## 📖 Table of Contents
TL,DR: ✅ marks the most important parts, scroll down to find them.
- [Introduction](#-introduction)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Integration](#integration)
- [Citation](#-citation)
- [License](#-license)

---
## 🔍 Introduction

ReasonCLIP is a CLIP-style training framework for improving visual representation learning with reasoning-aware supervision. This repository currently contains staged training recipes, evaluation pipelines, and a bundled llava_next workspace for downstream multimodal experiments.

## ⚡ Quick Start
### Quick Inference ✅

ReasonCLIP **does not modify any model architecture**. For inference/loading, please use the **official Hugging Face `transformers` code path**. You only need to replace the model ID with a ReasonCLIP checkpoint.

- Inference with ReasonCLIP or ReasonSigLIP
```python
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

model_id = "fesvhtr/RC-B32-S1"
model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

### Quick Evaluation

Evaluate one checkpoint with the standard benchmark suite:

```bash
bash scripts/eval_single.sh fesvhtr/RC-B32-S1
```

To reproduce the full released-model table, run the full sweep:

```bash
bash scripts/eval_all.sh
```

---

## 🚀 Training
### Before Training
#### Dataset Preparation ✅
All the training data are available at Huggingface. Detailed Dataset Card.
<details>
<summary>Click to expand data download code</summary>
</details>

#### Environment Preparation

### Stage 1

> [!NOTE]
> **To reproduce these results, use the `llava_next` folder contents.**

```bash
bash scripts/train_s1.sh
```

### Stage 2
```bash
bash scripts/train_s2.sh
```

### Direct Training (S0-Rea & S0-Des)

Descriptive supervision:

```bash
bash scripts/train_des_direct.sh
```

Reasoning supervision:

```bash
bash scripts/train_rea_direct.sh
```

---

## 📊 Evaluation

### Evaluate a Single Model

```bash
bash scripts/eval_single.sh fesvhtr/RC-B32-S1
```

Replace the argument with any checkpoint from the model table. Released checkpoints include their processor files, so no processor argument is required. To override the processor manually, pass it as the second argument.

This runs the standard evaluation suite for one checkpoint:

- ImageNet zero-shot classification
- Urban1k retrieval
- MSCOCO retrieval
- Flickr30k retrieval
- WinoGAViL
- compositional evaluation
- SugarCrepe++

### Full ReasonCLIP Evaluation Sweep

```bash
bash scripts/eval_all.sh
```

This script evaluates every released ReasonCLIP checkpoint listed in `model/models_final.sh`. `eval_all.sh` calls `scripts/eval_single.sh` once for each model. If a model list defines an optional `processors` array, those processor paths are used as overrides.

Use this only when reproducing the full table or benchmarking all released models. For normal use, evaluate a single checkpoint with `scripts/eval_single.sh`.

The full sweep covers the same benchmarks:

- ImageNet zero-shot classification
- Urban1k retrieval
- MSCOCO retrieval
- Flickr30k retrieval
- WinoGAViL
- compositional evaluation
- SugarCrepe++

### RCLIP Evaluation ✅

```bash
bash scripts/eval_rclip.sh
```

This script covers:

- RCLIP commonsense reasoning evaluation
- RCLIP retrieval evaluation

### Individual Evaluation Entrypoints

```bash
python eval/eval_zeroshot_imagenet.py --help
python eval/eval_retrieval.py --help
python eval/eval_winogavil.py --help
python eval/eval_sugarcrepe_pp.py --help
python eval/eval_RCLIP.py --help
```



---

## Integration

The repository also includes a `llava_next/` directory for downstream multimodal work.

`TODO: explain whether this is used for:`

- probing the learned encoder
- multimodal fine-tuning
- instruction tuning
- transfer evaluation

If you do not want to expose this yet, keep this section short and say it is an experimental downstream workspace.

All evaluations were conducted using [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

---

## 📝 Citation


---

## 📖 License

`TODO: add the repository license here.`

If you use external datasets, pretrained checkpoints, or bundled third-party code, also mention that users must comply with their original licenses.
