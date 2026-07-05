# ReasonCLIP model-centric reasoning explorer v3 报告

## 目标

v3 版本把分析主线从“对齐外部 caption/Qwen reference”改成“直接观察模型自身的 retrieval 和 image-neighborhood 变化”。

核心问题是：给模型一个只包含推理结论的 concept，例如 `could hurt if touched`、`may roll away`、`likely exposed to heat`，不在网页上展示也不在 concept label 里强调具体视觉证据；实际 retrieval prompt 可以是完整自然语言句子，但页面只显示短 label。然后比较 baseline / S1 / S2 是否能把推理结论连接到图像中的不同视觉证据。

## 数据和模型

- 数据：DOCCI 5K images。
- 模型：CLIP-L/14-224 baseline / S1 / S2，SigLIP-So400M/14-384 baseline / S1 / S2。
- 不使用 Qwen、MLLM 或 caption embedding 作为主 reference。
- DOCCI caption 只作为结果解释用的 source caption / snippet。

## 新版 explorer

线上地址：

https://fesvhtr.github.io/zsc/reasonclip/

本地产物：

- `reasoning_geometry/explorer_v3/index.html`
- `reasoning_geometry/explorer_v3/data/explorer_data.json`
- `reasoning_geometry/explorer_v3/data/explorer_data.js`
- `reasoning_geometry/explorer_v3/assets/thumbs/`

对应部署提交：

- `zsc` commit `6380abd`: `update ReasonCLIP explorer v3`
- `zsc` commit `d3a78f6`: `fix: avoid duplicate pages deploy`

## 页面结构

### 1. Inference Retrieval

右侧区域现在使用 tab 切换，默认显示 Inference Retrieval，另外两个 tab 是 Anchor Neighbor Compare 和 Neighbor Changed。

固定一个 inference-only concept，同一 family 下三列比较 baseline / S1 / S2。

每张卡片直接显示：

- rank
- similarity
- baseline rank / baseline similarity
- new/shared 标记
- rank delta
- DOCCI caption snippet

网页只显示短 label，例如 `may roll away`；完整 prompt 只保留在数据里用于追溯和复现。

### 2. Anchor Neighbor Compare

点击任意图片后，同一 family 下三列展示该 anchor 在 baseline / S1 / S2 image space 里的 top neighbors。

每个 neighbor 显示 rank 和 cosine similarity，并标记 shared / new。

### 3. Neighbor Changed

对同一个 anchor，展示 S1/S2 相比 baseline 的三类变化：

- Pulled closer：S1/S2 新拉近或 rank 明显提前的图。
- Pushed away：baseline 里靠前但在 S1/S2 中被推远的图。
- Stable shared：baseline 和 S1/S2 都保留的邻居。

每个 changed item 显示 baseline rank/similarity、stage rank/similarity 和 delta。

## 摘要指标

这些指标只说明分布变化幅度，不直接等价于 reasoning 正确率。

### Retrieval top-30 与 baseline 的重合

| family | stage | avg shared | avg new | max positive rank delta |
| --- | ---: | ---: | ---: | ---: |
| CLIP-L/14-224 | S1 | 11.80 | 18.20 | 2606 |
| CLIP-L/14-224 | S2 | 7.40 | 22.60 | 4858 |
| SigLIP-So400M | S1 | 6.27 | 23.73 | 3840 |
| SigLIP-So400M | S2 | 5.20 | 24.80 | 3657 |

解释：S1/S2 的 inference-only retrieval top-30 和 baseline 有大量不同结果，说明训练后 text-to-image retrieval 分布确实发生了明显变化。S2 通常比 S1 更激进，尤其在 CLIP family 中更明显。

### Anchor top-12 neighbor 与 baseline 的重合

| family | stage | avg top-12 overlap | avg new neighbors |
| --- | ---: | ---: | ---: |
| CLIP-L/14-224 | S1 | 7.32 | 4.68 |
| CLIP-L/14-224 | S2 | 5.68 | 6.32 |
| SigLIP-So400M | S1 | 7.62 | 4.38 |
| SigLIP-So400M | S2 | 6.60 | 5.40 |

解释：image-neighborhood 也明显重排，但不是完全替换。S1 更保守，保留更多 baseline 邻域；S2 更激进，产生更多 new neighbors。这和我们想观察的 S1/S2 差异一致。

## 目前结论

v3 页面已经能直接展示三个层面的证据：

- 同一个 inference-only concept 下，S1/S2 相比 baseline 检索到哪些新图。
- 同一个 anchor 下，S1/S2 的 image nearest neighbors 如何变化。
- 哪些图被拉近、哪些图被推远、哪些基础邻居仍然稳定保留。

这个版本不声称自动证明模型拥有完整 commonsense reasoning。它的作用是把“训练后分布是否更 reasoning-aware”变成可逐例检查的东西：如果 S1/S2 新拉近的图跨 object 但共享风险、后果、功能或物理状态，就支持 visually grounded commonsense reasoning alignment；如果只是 object/attribute shortcut，就在页面上可以直接看出来。

## 验证

已完成检查：

- 本地 `explorer_v3`：5000 records，6 models，15 concepts。
- 所有 image paths 都是 `assets/thumbs/...` 相对路径，无本地绝对路径泄漏。
- `index.html` 不显示 `things that`、`retrieval_prompt`、Qwen 或旧 caption-reference 指标。
- retrieval、neighbors、neighbor_changes 都包含 rank 和 similarity。
- 本地静态服务器检查：HTML、data JS、缩略图均返回 200。
- GitHub Actions：`Build site` success，`pages build and deployment` success。
- 线上检查：`https://fesvhtr.github.io/zsc/reasonclip/` 返回 200，HTML 标题为 `ReasonCLIP Model-Centric Explorer`，数据 schema 为 `v3`。
