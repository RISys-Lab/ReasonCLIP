# ReasonCLIP 5K Reasoning Geometry 分析报告

## 结论摘要

这次分析用 DOCCI 5K 图片做了正式版表征空间分析，目标是看 ReasonCLIP 是否不仅保持基础视觉能力，还能让 visually grounded commonsense reasoning 更好地对齐到 image embedding space。

主要结论：

- **CLIP-L/14-224 系列：ReasonCLIP 的 reasoning alignment 明显增强。** S1 和 S2 的 caption-neighborhood overlap 与 RSA 都高于 baseline；S1 同时提升 caption-neighbor image similarity 和 triplet margin，更像是在保留原始视觉结构的同时加入 reasoning sensitivity；S2 的 RSA 最高，但局部 similarity/margin 不如 S1，说明 S2 对全局几何结构改动更强。
- **SigLIP-So400M 系列：局部 reasoning 近邻更强，但全局 RSA 下降。** S1/S2 的 caption-neighbor image similarity 和 triplet margin 都明显优于 baseline，说明相似 reasoning caption 对应的图片更容易被拉近；但 similarity-matrix RSA 低于 baseline，说明 SigLIP 的全局 pairwise geometry 没有简单变成 Qwen caption space 的同构结构。
- **这个分析支持“reasoning alignment 有改善”，但不同 backbone 的改善形态不同。** CLIP 更像全局和局部都往 reasoning space 靠；SigLIP 更像局部 reasoning cluster/positive pair 改善，而全局结构仍保留或重排了别的视觉因素。

## 数据和标注

- 数据：DOCCI 本地 5K 子集。
- 图片数：5,000。
- reasoning captions：14,931 条，约每张 3 条。
- 标注方法：`docci_text_rules_v2_evidence_first`。
- 标注来源：DOCCI long caption 规则拆解，不包装成人类 ground truth。
- caption semantic reference：`Qwen/Qwen3-1.7B` text-only hidden states，mean pooling，L2 normalize。

没有使用 LLaVA，也没有使用 CLIP/ReasonCLIP 自己的 text encoder 来构造 reasoning reference，避免循环论证。

## 模型

| family | baseline | S1 | S2 |
|---|---|---|---|
| CLIP-L/14-224 | `openai/clip-vit-large-patch14` | `fesvhtr/clip-r-s1-run1207-1280` | `fesvhtr/clip-r-s2-run1219-505` |
| SigLIP-So400M/14-384 | `google/siglip-so400m-patch14-384` | `fesvhtr/siglip-r-s1-run0201-1280` | `fesvhtr/siglip-r-s2-run0203-673` |

S2 checkpoint 里的 classification heads 在 inference 加载时被丢弃，这是训练阶段 head 的预期行为。

## 正式指标

| model | neighbor@10 | caption-neighbor image sim | RSA Spearman | triplet margin |
|---|---:|---:|---:|---:|
| clip_base | 0.06192 | 0.59619 | 0.19201 | -0.09697 |
| clip_s1 | 0.07028 | 0.65237 | 0.26526 | -0.07352 |
| clip_s2 | 0.07342 | 0.51433 | 0.29028 | -0.10069 |
| siglip_base | 0.06674 | 0.58390 | 0.16943 | -0.09216 |
| siglip_s1 | 0.06726 | 0.67784 | 0.10469 | -0.06401 |
| siglip_s2 | 0.06900 | 0.66459 | 0.08833 | -0.06387 |

相对 baseline 的变化：

| family | stage | neighbor@10 delta | caption-neighbor sim delta | RSA delta | triplet margin delta |
|---|---|---:|---:|---:|---:|
| CLIP-L/14-224 | S1 | +0.00836 | +0.05618 | +0.07325 | +0.02345 |
| CLIP-L/14-224 | S2 | +0.01150 | -0.08186 | +0.09827 | -0.00372 |
| SigLIP-So400M | S1 | +0.00052 | +0.09394 | -0.06475 | +0.02815 |
| SigLIP-So400M | S2 | +0.00226 | +0.08069 | -0.08110 | +0.02828 |

## 指标解释

- `neighbor@10`：caption semantic space 中 top-10 reasoning 近邻，和 image embedding space top-10 近邻的 overlap。越高表示 reasoning-neighborhood 更一致。
- `caption-neighbor image sim`：caption-neighbor pairs 在 image embedding space 里的平均 cosine similarity。越高表示 reasoning 相似的图片在视觉 embedding 中更近。这个值受模型整体 similarity scale 影响，主要在同一 family 内比较。
- `RSA Spearman`：caption-caption similarity matrix 和 image-image similarity matrix 的 sampled Spearman correlation。越高表示全局 pairwise geometry 更像 reasoning caption space。
- `triplet margin`：caption positive 与 image-hard-negative 的 image similarity 差值。这里绝对值仍为负，说明“视觉上很像但 reasoning 不像”的 hard negative 仍然很强；delta 变大表示模型更能把 reasoning positive 拉近。

## Interpretation

CLIP 系列里，S1 是最稳的版本：它在 local positive similarity、RSA、triplet margin 三个方向都比 baseline 好。S2 的全局 RSA 最强，neighbor overlap 也最高，但 caption-neighbor absolute similarity 和 triplet margin 不如 S1，说明 S2 可能更强地重塑了全局结构，而不是单纯把所有 reasoning-positive pairs 拉近。

SigLIP 系列里，S1/S2 都明显提升了 caption-neighbor image similarity 和 triplet margin，说明 reasoning 相似图片的局部拉近是存在的。但 RSA 下降说明它们的全局 similarity matrix 没有更像 Qwen caption semantic matrix。这个结果不一定是坏事：SigLIP 原始空间可能已经有更强的视觉/语义全局结构，ReasonCLIP training 后把局部 reasoning-positive 拉近，同时全局 pairwise ordering 变得不像 text-only caption space。

## 可视化产物

最终 explorer：

```text
/home/localadmin/bz/ReasonCLIP/reasoning_geometry/explorer/index.html
```

数据文件：

```text
/home/localadmin/bz/ReasonCLIP/reasoning_geometry/explorer/data/explorer_data.js
```

Explorer 内容：

- 5K image embedding scatter。
- baseline / S1 / S2 模型选择。
- reasoning type / split 着色。
- 8 个 reasoning prompts 的 text-to-image retrieval，对比 baseline、S1、S2。
- 点击图片查看 source caption、reasoning caption、同模型 image nearest neighbors。

## 产物位置

```text
reasoning_geometry/work/v2/annotations.jsonl
reasoning_geometry/work/v2/caption_embeddings_qwen/embeddings.npy
reasoning_geometry/work/v2/image_embeddings/*.npy
reasoning_geometry/work/v2/metrics.json
reasoning_geometry/work/v2/retrievals.json
reasoning_geometry/explorer/index.html
```

V2 产物大小：`reasoning_geometry/work/v2` 约 173MB，`reasoning_geometry/explorer` 约 22MB。

## 限制

- reasoning annotations 是从 DOCCI long captions 自动拆解出来的，不是人工标注 ground truth。
- caption reference 使用 Qwen text-only LM hidden states，不是专门训练的 embedding 模型；BGE 下载太慢，未作为正式结果使用。
- 这次分析是 representation geometry 分析，不替代 ImageNet/retrieval benchmark。基础能力是否提升仍以已有 benchmark 为主；这里主要看 reasoning concept 是否在 image space 中更 align。
- PCA scatter 是可视化辅助，不作为定量结论来源。定量结论以 metrics 和 qualitative retrieval inspection 结合判断。
