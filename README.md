<div align="center">
  <img src="asset/reasonclip_wordmark_compact.svg" alt="ReasonCLIP" width="100%">

📄 **[Paper](https://img.shields.io/badge/Paper-TODO-gray)** |
🤗 **[Models](https://img.shields.io/badge/Models-TODO-gray)** |
🤗 **[Dataset](https://img.shields.io/badge/Dataset-TODO-gray)** |
🤗 **[BenchMark](https://img.shields.io/badge/Dataset-TODO-gray)** |
📊 **[Model Card](https://img.shields.io/badge/Models-TODO-gray)**
</div>

## News

- `[TODO date]` Add project release note here.
- `[TODO date]` Add model / dataset / code update here.
- `[TODO date]` Add paper / benchmark / checkpoint update here.


<details>
<summary>More</summary>

- `[TODO older date]` Older update 1.
- `[TODO older date]` Older update 2.

</details>

---
## 📖 Table of Contents
TL,DR: ✅ marks the most important parts, scroll down to find them.
- Introduction
- Quick Start
- Training
- Evaluation
- LLaVA-NeXT Integration
- Citation
- License

---
## 🔍 Introduction

ReasonCLIP is a CLIP-style training framework for improving visual representation learning with reasoning-aware supervision. This repository currently contains staged training recipes, direct-training variants, evaluation pipelines, and a bundled `llava_next/` workspace for downstream multimodal experiments.



## ⚡ Quick Start

### Quick Inference ✅

ReasonCLIP **does not modify any model architecture**. For inference/loading, please use the **official Hugging Face `transformers` code path**. You only need to replace the model ID with a ReasonCLIP checkpoint.

- Inference with ReasonCLIP
```python
from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("fesvhtr/ReasonSigLIP-so400m-patch14-384-S2")
processor = AutoProcessor.from_pretrained("fesvhtr/ReasonSigLIP-so400m-patch14-384-S2")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

### Quick Evaluation

Run the bundled evaluation sweep:

```bash
bash scripts/eval_all.sh
```

Or run the reasoning benchmark only:

```bash
bash scripts/eval_rclip.sh
```


---

## 🚀 Training

`TODO: add your final training recipe overview here.`

Suggested way to write this section:

- Stage 1: what supervision is used and what is initialized
- Stage 2: what extra heads / objectives / losses are introduced
- Direct variants: when to use `des_direct` vs `rea_direct`
- CLIP vs SigLIP switching notes

### Dataset Preparation ✅
All the training data are available at Huggingface. Detailed Dataset Card.
<details>
<summary>Click to expand data download code</summary>
</details>


### Stage 1

```bash
bash scripts/train_s1.sh
```
<details>
<summary>Click to expand Stage 1 Training scripts</summary>
</details>

### Stage 2

```bash
bash scripts/train_s2.sh
```
<details>
<summary>Click to expand Stage 2 Training scripts</summary>
</details>

### Direct Training (S0-Rea & S0-Des)

Descriptive supervision:

```bash
bash scripts/train_des_direct.sh
```
<details>
<summary>Click to expand Stage 0 - Descriptive Training scripts</summary>
</details>

Reasoning supervision:

```bash
bash scripts/train_rea_direct.sh
```
<details>
<summary>Click to expand Stage 0 - Reasoning Training scripts</summary>
</details>

---

## 📊 Evaluation

### Full Evaluation Sweep

```bash
bash scripts/eval_all.sh
```

This script covers these benchmarks:

- ImageNet zero-shot classification
- Urban1k retrieval
- MSCOCO retrieval
- Flickr30k retrieval
- WinoGAViL
- compositional evaluation
- SugarCrepe++

This script covers these models:

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

### Placeholder Result Table

`TODO: replace with your final numbers.`

| Model | ImageNet | Urban1k | Flickr30k | WinoGAViL | SugarCrepe++ | RCLIP |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline CLIP | TODO | TODO | TODO | TODO | TODO | TODO |
| ReasonCLIP | TODO | TODO | TODO | TODO | TODO | TODO |
| ReasonCLIP + variant | TODO | TODO | TODO | TODO | TODO | TODO |




---

## LLaVA-NeXT Integration and Evaluation

The repository also includes a `llava_next/` directory for downstream multimodal work.

`TODO: explain whether this is used for:`

- probing the learned encoder
- multimodal fine-tuning
- instruction tuning
- transfer evaluation

If you do not want to expose this yet, keep this section short and say it is an experimental downstream workspace.


---

## 📝 Citation

`TODO: replace with your actual BibTeX.`

```bibtex
@article{reasonclip_todo_2026,
  title={ReasonCLIP: TODO},
  author={TODO},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```

If this repo includes multiple papers or technical reports, you can add them all here.

---

## 📖 License

`TODO: add the repository license here.`

If you use external datasets, pretrained checkpoints, or bundled third-party code, also mention that users must comply with their original licenses.
