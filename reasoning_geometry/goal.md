# ReasonCLIP 视觉 grounded commonsense reasoning 分布分析目标 v2

环境优先使用 `/home/localadmin/venvs/llm`。不要为了这个分析改 torch / vLLM 等核心依赖版本；普通数据处理、可视化、静态网页依赖可以按需安装。

## 核心问题

我们不只想证明 ReasonCLIP 在 benchmark 上分数更高，还想看训练后的 CLIP / SigLIP 表征是否真的更像“视觉证据支持的常识推理”。

这里的 reasoning 不是简单 attribute，也不是只检索证据词。我们关心的是：

> 给模型一个推理结论、风险、后果、用途或隐含状态，它能不能自己把它连接到图像里的可见证据？

例如：

- 页面显示 `could hurt if touched`，retrieval prompt 可用 `things that could hurt someone if touched` -> 可能找破玻璃、尖锐工具、火、裸露电线等。
- 页面显示 `may roll away`，retrieval prompt 可用 `things that may roll away` -> 可能找球、轮子、圆柱、斜坡上的圆形物。
- 页面显示 `likely exposed to heat`，retrieval prompt 可用 `things that are likely exposed to heat` -> 可能找融化冰激凌、融化蜡、冒热气的食物、晒化的东西。
- 页面显示 `may fall soon`，retrieval prompt 可用 `things that may fall soon` -> 可能找悬空、倾斜、堆叠不稳、边缘附近的物体。

也就是说，prompt 里不直接写 `broken glass`、`round ball`、`melting ice cream` 这种视觉 evidence。证据应该出现在 retrieval 结果里，而不是 query 里。

最终要回答：

- Baseline CLIP / SigLIP 是否主要按 object、scene、attribute 相似来组织图片？
- S1 / S2 是否更容易把共享 commonsense inference 的图片拉近，即使它们的 object 不同？
- S1 / S2 是否仍保留原始视觉能力，而不是只把分布改乱？
- S1 和 S2 的差异是什么：S1 是否更稳，S2 是否更强地重排局部邻域？

## 关键原则

### 1. 不再使用 Qwen / MLLM 作为主 reference

新的主分析不使用 Qwen caption embedding、MLLM caption embedding、或任何外部 language model semantic space 作为“reasoning reference”。

原因：

- 我们现在想看的是模型自身的 image/text embedding 行为，而不是它是否贴近另一个 LLM 的文本语义空间。
- 如果用外部 caption embedding 做 reference，解释会变成“ReasonCLIP 是否更像 Qwen 的 caption space”，这不是当前目标。
- DOCCI / Visual Genome 的文字可以用来辅助读图、显示原始描述、筛选样本，但不作为主指标的语义空间。

旧版本中的 caption-neighborhood alignment、RSA against caption embeddings、caption-triplet metrics 全部降级为历史附录或删除，不放在新版 explorer 的主界面。

### 2. Query 必须是 inference-only

检索 prompt 应该只写推理结论、后果、风险、功能、隐含状态，不写直接视觉证据。

不推荐：

- `broken glass that could injure someone`
- `round objects that may roll away`
- `melting ice cream exposed to heat`
- `objects leaning and likely to fall`

推荐把网页展示 label 和实际 retrieval prompt 分开：

- display: `could hurt if touched`; retrieval: `things that could hurt someone if touched`
- display: `may roll away`; retrieval: `things that may roll away`
- display: `likely exposed to heat`; retrieval: `things that are likely exposed to heat`
- display: `may fall soon`; retrieval: `things that may fall soon`

这样才能测试模型是否能从 inference 反推到多种视觉 evidence，而不是只靠 object / attribute noun retrieval。

### 3. Evidence 只在结果解释中展示

结果卡片可以显示 DOCCI 原 caption 或自动摘出的短 evidence，帮助人读结果。但这些 evidence 不参与 query。

目标展示方式：

```text
Concept: could hurt if touched
Hidden retrieval prompt: things that could hurt someone if touched

Retrieved image:
  image
  score
  caption/evidence snippet: "shattered glass is scattered on the ground ..."
```

这样读者可以看到模型是否真的把“could hurt someone”连到了“shattered glass / sharp object / fire / exposed wire”这类视觉证据。

### 4. 所有结果都必须显示 similarity

新版 explorer 中，retrieval results、anchor neighbors、pulled closer、pushed away、stable shared 都必须直接显示 rank 和 similarity。

要求：

- similarity 不要藏在 hover 或被卡片宽度截断。
- retrieval 卡片显示 `rank`、`similarity`、可选 `rank delta` / `similarity delta`。
- image-neighbor 卡片显示相对 anchor 的 cosine similarity。
- neighbor changed 卡片显示 baseline 与 S1/S2 的 similarity 对比和 delta。
- 数字可以小，但必须稳定可见，建议独立一行显示，例如 `#03 · sim 0.2471`。

## 数据选择

优先继续使用 DOCCI；Visual Genome 作为备用或补充。

### DOCCI

用途：

- 提供 5K 级别图片和 human-written long captions。
- 原 caption 用于人读结果、检查 retrieval 是否合理。
- 可以从 caption 中自动抽取短 evidence snippet，用于结果卡片展示。

注意：

- DOCCI caption 不作为主 reasoning reference。
- 不再把 DOCCI caption 规则模板化成 reasoning caption 来做主指标。

### Visual Genome

用途：

- 如需更明确的 object / attribute / relationship 辅助，可以用 VG 补充 anchor 或 hard cases。
- VG 的 objects / relationships 可以用来筛出场景类型，但不作为最终 reasoning ground truth。

## 模型范围

主模型仍然比较两个 backbone，每个 backbone 分 baseline / S1 / S2。

### CLIP-L/14-224

- Baseline: `openai/clip-vit-large-patch14`
- Stage 1: `fesvhtr/clip-r-s1-run1207-1280`
- Stage 2: `fesvhtr/clip-r-s2-run1219-505`

### SigLIP1-So400M/14-384

- Baseline: `google/siglip-so400m-patch14-384`
- Stage 1: `fesvhtr/siglip-r-s1-run0201-1280`
- Stage 2: `fesvhtr/siglip-r-s2-run0203-673`

CLIP 和 SigLIP 分开比较。不同 backbone 的 embedding scale、训练目标和 text encoder 行为不同，不直接混在同一个结论里。

## 新版 Explorer 目标

新版可视化要让用户不需要读复杂指标，就能看到：

- 同一个 inference-only query 下，baseline / S1 / S2 retrieve 的结果如何变化。
- 同一张 anchor image 下，baseline / S1 / S2 的 image nearest neighbors 如何变化。
- S1 / S2 相比 baseline 新拉近了哪些图片，又推远了哪些图片。

### 1. Embedding Space

保留左侧全局 embedding space 视图。

每个点是一张图片；选择模型后展示对应 image embedding 的 2D projection。

改动：

- 不再用 Qwen/caption cluster 着色。
- 可以支持按 split、selected query score、selected anchor neighborhood、model stage delta 着色。
- 点击点后进入 anchor neighbor 对照。

这个视图只作为整体分布感知，不作为主要证明。

### 2. Inference-only Retrieval

这是新版主视图之一。

固定一个 inference-only concept，然后同一 family 内三列展示。实现时每个 concept 至少有两个字段：

- `display_label`：网页上显示的短词语或短语，例如 `may roll away`。
- `retrieval_prompt`：真正送进模型 text encoder 的完整句子，例如 `things that may roll away`。

网页默认只显示 `display_label`，不要把 `things that` 这类模板前后缀暴露给用户。检索时仍然可以使用完整自然语言 prompt。

然后同一 family 内三列展示：

- baseline
- S1
- S2

每张卡片展示：

- image
- model score / rank
- DOCCI source caption 或短 evidence snippet
- 是否是 `new in S1/S2`
- 是否与 baseline shared
- rank delta：相比 baseline 提前了多少

核心问题：

> S1/S2 是否能从同一句抽象推理结论中找出更合理、更多样的视觉证据？

候选 concept 初版：

| display label | retrieval prompt |
| --- | --- |
| `could hurt if touched` | `things that could hurt someone if touched` |
| `may roll away` | `things that may roll away` |
| `may fall soon` | `things that may fall soon` |
| `may spill or leak` | `things that may spill or leak` |
| `likely exposed to heat` | `things that are likely exposed to heat` |
| `may be slippery` | `things that may be slippery` |
| `protects from danger` | `things that may protect someone from danger` |
| `may no longer work` | `things that may no longer work properly` |
| `may break under pressure` | `things that may break under pressure` |
| `hard to balance` | `things that are hard to keep balanced` |
| `could cause loss of balance` | `things that could make someone lose balance` |
| `unsafe for a child` | `things that may be unsafe for a child` |
| `prevents damage` | `things that are likely being used to prevent damage` |
| `changed shape by force` | `things that may have changed shape because of force` |
| `needs cleanup` | `things that may need to be cleaned up` |

Concept 迭代原则：

- 少写或不写具体 object noun。
- 少写直接 evidence word。
- 尽量写后果、风险、功能、隐含物理状态。
- 如果 baseline 也能轻易检索出来，说明 prompt 可能太 evidence-heavy，需要重写。

### 3. Anchor Neighbor Compare

这是新版第二个主视图。

固定一张 anchor image，然后同一 family 内三列展示它的 nearest image neighbors：

- baseline image space
- S1 image space
- S2 image space

每个 neighbor 显示：

- image
- rank / similarity
- source caption snippet
- shared / new / dropped 标记

核心问题：

> 同一张图在训练前后，邻域是否从 object/scene 相似，变成更关注共享推理后果或隐含状态？

例子：

- anchor 是融化冰激凌：baseline 可能找甜品/食物；S1/S2 如果更好，可能找融化、热、液体化、变形的其他物体。
- anchor 是破玻璃：baseline 可能找窗户/玻璃；S1/S2 如果更好，可能找尖锐、危险、损坏、可能伤人的视觉证据。
- anchor 是圆球在边缘：baseline 可能找球；S1/S2 如果更好，可能找会滚、难平衡、可能掉落的不同物体。

### 4. Neighbor Changed View

这是新版第三个主视图，也是最能说明“分布变了”的部分。

对每个 anchor，比较 S1/S2 相比 baseline 的 neighbor rank change：

#### Pulled closer

S1/S2 中新进入 top-k，或 rank 明显提前的图片。

显示：

```text
image
rank_baseline -> rank_s1
similarity_baseline -> similarity_s1
caption snippet
```

要观察：

- 是否共享 inference/consequence，而不是只共享 object。
- 是否跨 object 但 reasoning 状态相似。

#### Pushed away

baseline top-k 里靠前，但在 S1/S2 中 rank 明显下降的图片。

要观察：

- 是否只是 object/scene 相似，但 reasoning 状态不相似。
- S1/S2 是否减少了这类 object-only neighbor。

#### Stable shared

baseline 和 S1/S2 都保留的邻居。

用途：

- 检查基础视觉能力是否保留。
- 如果合理的 object/scene neighbor 仍然保留，说明模型不是简单破坏原始空间。

### 5. Case Gallery

从 inference-only retrieval 和 anchor neighbor compare 中挑典型案例，形成固定 gallery：

- clear win：S1/S2 明显更符合推理结论。
- baseline already good：baseline 本来就能做，说明这个 query 不够有区分度。
- failure：S1/S2 被语言先验或 object bias 带偏。
- S1 vs S2 difference：S1 更稳或 S2 更激进的例子。

这个 gallery 用于最终报告，不只依赖随机点选。

## 定量指标重新定义

新版主指标不再依赖外部 caption embedding reference。

### 1. Retrieval overlap / rank-change

对每个 inference-only query，在同一 family 内计算：

- top-k overlap：baseline vs S1，baseline vs S2，S1 vs S2。
- new-in-stage count：S1/S2 相比 baseline 新进入 top-k 的图片数。
- rank delta：S1/S2 中提前最多的图片。

这不是“正确率”，而是量化分布变化程度。

### 2. Anchor neighbor overlap / rank-change

对每个 anchor image：

- baseline top-k 与 S1/S2 top-k overlap。
- S1/S2 pulled closer list。
- S1/S2 pushed away list。
- shared stable neighbors。

这能量化局部邻域是否重排。

### 3. Base capability sanity check

为了避免“reasoning 看起来变了但基础能力坏了”，保留基础检查：

- 现有 benchmark 结果作为主依据。
- 在 explorer 中看 stable shared neighbors 是否仍有合理 object/scene 相似。
- 对 descriptive/object-like prompts 做少量 sanity retrieval，确认不是只会抽象推理、不认物体。

### 4. 小规模人工/模型判读

如果需要给 retrieval 结果打分，只对少量 query 的 top-k 做人工式判读：

- inference 是否合理。
- visible evidence 是否支持该 inference。
- 是否只是 object noun shortcut。

这可以由我们自己逐例判断，不包装成大规模自动 ground truth。

## 分版本推进

### V0: 改 explorer 跑通

使用现有 5K image embeddings，不重新跑大规模 embedding。

目标：

- 删除 Qwen/caption reference 主指标展示。
- 加入 inference-only prompt 列表。
- 右上角 retrieval 改成更明确的 baseline / S1 / S2 三列。
- 每张 retrieval 卡片显示 rank、score、caption snippet、new/shared 标记。

产出：

- `retrievals_inference_only.json`
- `explorer_v3/`
- 一批初步截图和观察。

### V1: Anchor neighbor compare

目标：

- 点击任意图片后显示 baseline / S1 / S2 三列 nearest neighbors。
- 支持 CLIP family 和 SigLIP family 切换。
- 显示 shared / new / dropped。

产出：

- `neighbor_compare.json`
- explorer 中的 anchor compare panel。

### V2: Neighbor changed 正式版

目标：

- 对每个 anchor 预计算 S1/S2 vs baseline 的 pulled closer / pushed away / stable shared。
- 加入 Neighbor Changed View。
- 支持按 rank delta 和 similarity delta 排序。

产出：

- `neighbor_changes.json`
- explorer 中的 changed view。
- 典型 case gallery。

### V3: Formal report

目标：

- 基于 5K DOCCI 正式版写报告。
- 按 CLIP 和 SigLIP 分开分析 baseline / S1 / S2。
- 重点展示 inference-only retrieval、anchor neighbor compare、neighbor changed 三类证据。

报告结论不要过度自动化：

- 可以说“这些 case 和 rank-change 显示 S1/S2 更倾向于拉近共享推理后果的图片”。
- 不说“模型被证明拥有某种完整 commonsense reasoning 能力”。
- 对失败案例和 baseline already good 案例也要展示。

## 最终产物

- 新版静态 explorer，部署到 `https://fesvhtr.github.io/zsc/reasonclip/` 或其子路径。
- inference-only retrieval prompts。
- baseline / S1 / S2 retrieval 对比结果。
- anchor neighbor compare 结果。
- neighbor changed 结果。
- 简短中文报告，解释哪些变化支持 ReasonCLIP 的 visually grounded commonsense reasoning alignment，哪些变化只是 object/attribute retrieval。

## 成功标准

新版分析如果有效，用户应该能直接看到：

- 同一个 inference-only query 下，S1/S2 相比 baseline 找到更多“视觉证据不同但推理结论相同”的图片。
- 同一个 anchor 下，S1/S2 新拉近的图片不只是同物体，而是共享风险、后果、物理状态、功能或隐含原因。
- S1/S2 仍保留一部分合理 object/scene neighbor，说明基础能力没有明显崩。
- CLIP 与 SigLIP 的 S1/S2 行为差异能通过具体案例解释，而不是只看一个整体分数。
