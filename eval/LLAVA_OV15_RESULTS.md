# LLaVA-OneVision 1.5 本地评测总结

整理日期：2026-10-25。汇总本地 `eval_results/llava_ov15/` 下的 34 份 `*_results.json`，结果记录日期为 2026-07-30 至 2026-08-07。三个模型各有 11 项通用任务和 15 项推理任务的完整聚合结果。本文指标直接提取自已保存的 JSON 结果。

`reasoning_geometry/report_v2.md`、`report_v3.md` 是独立的图像表征分析报告；本文件记录 LLaVA 多模态模型评测。

## 模型对应关系

三个模型均采用 LLaVA-OneVision 1.5 架构和 Qwen3-8B 语言模型，下表对应的都是完成 `stage2_instruct` 后的模型。

| 本文名称 | 原本地目录（位于 `outputs/llava_ov15/`） | 视觉编码器 |
| --- | --- | --- |
| SigLIP 基线 | `siglip1/stage2_instruct` | `google/siglip-so400m-patch14-384` |
| ReasonSigLIP S1 | `rs1-1/stage2_instruct` | `RISys-Lab/ReasonSigLIP-So14-384-S1` |
| ReasonSigLIP S2 | `reasonsiglip/stage2_instruct` | `RISys-Lab/ReasonSigLIP-So14-384-S2` |

推理结果目录中的 `rs1-2` 实际加载 `reasonsiglip/stage2_instruct`，对应 ReasonSigLIP S2。这里 S1 / S2 指视觉编码器的训练版本，与 LLaVA 自身的 stage1 / stage2 训练流程不同。

## 通用评测

除 MME 两项为原始分数外，其余均显示为百分数，越高越好。每个数值链接到对应原始结果 JSON。MMVP accuracy 与 pair accuracy 使用同一批 300 个问题，后者按配对计分；AI2D 两种设置也分别列出，不作为独立数据集累加。

OK-VQA 报告本地 `exact_match` 指标，对应结果文件的 `submission` 字段均为 `null`。

| 任务 / 指标 | 样本数 | SigLIP 基线 | ReasonSigLIP S1 | ReasonSigLIP S2 |
| --- | ---: | ---: | ---: | ---: |
| AI2D | 3088 | [80.25][R34] | [80.38][R33] | [79.57][R32] |
| AI2D no mask | 3088 | [89.93][R34] | [90.19][R33] | [89.44][R32] |
| ChartQA overall | 2500 | [67.28][R34] | [67.28][R33] | [67.16][R32] |
| GQA | 12578 | [64.71][R34] | [64.55][R33] | [64.86][R32] |
| MME cognition（原始分） | 2374 | [358.93][R34] | [375.00][R33] | [360.36][R32] |
| MME perception（原始分） | 2374 | [1659.53][R34] | [1657.27][R33] | [1646.18][R32] |
| MMStar average | 1500 | [56.10][R34] | [55.10][R33] | [56.40][R32] |
| MMVP accuracy | 300 | [79.00][R34] | [77.00][R33] | [77.67][R32] |
| MMVP pair accuracy | 300 | [58.00][R34] | [56.00][R33] | [58.00][R32] |
| OK-VQA val2014 exact match | 5046 | [43.64][R34] | [53.66][R33] | [37.37][R32] |
| RealWorldQA | 765 | [65.36][R34] | [64.84][R33] | [64.44][R32] |
| ScienceQA | 4241 | [85.55][R34] | [85.95][R33] | [85.10][R32] |
| VisuLogic | 1000 | [27.60][R34] | [24.90][R33] | [25.30][R32] |

## 推理与空间评测

所有分数显示为百分数，差值单位为百分点，越高越好。MathVision 原始 JSON 已经是百分制，直接保留；其他任务的 0–1 数值乘以 100。`spatial457` 分组占位行没有数值指标，保留其七个子任务。

| 任务 / 指标 | 样本数 | SigLIP 基线 | ReasonSigLIP S1 | ReasonSigLIP S2 | S1 − 基线 | S2 − 基线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VisualPuzzles direct | 1168 | [30.65][R31] | [30.14][R01] | [30.05][R16] | -0.51 | -0.60 |
| BLINK IQ test | 150 | [28.00][R31] | [28.00][R02] | [26.00][R17] | +0.00 | -2.00 |
| BLINK jigsaw | 150 | [52.67][R31] | [52.67][R03] | [52.67][R18] | +0.00 | +0.00 |
| BLINK multi-view reasoning | 133 | [43.61][R31] | [54.14][R04] | [55.64][R19] | +10.53 | +12.03 |
| CoreCognition overall | 1423 | [56.36][R31] | [57.48][R05] | [56.85][R20] | +1.12 | +0.49 |
| CV-Bench | 2638 | [74.26][R31] | [73.16][R06] | [74.15][R21] | -1.10 | -0.11 |
| MathVision testmini | 304 | [19.74][R31] | [21.05][R07] | [21.71][R22] | +1.31 | +1.97 |
| Spatial457 L1 single | 670 | [42.39][R31] | [47.01][R08] | [42.24][R23] | +4.63 | -0.15 |
| Spatial457 L2 objects | 4995 | [40.54][R31] | [40.36][R09] | [36.64][R24] | -0.18 | -3.90 |
| Spatial457 L3 2D spatial | 4995 | [38.38][R31] | [38.32][R10] | [35.16][R25] | -0.06 | -3.22 |
| Spatial457 L4 occlusion | 3534 | [20.74][R31] | [23.91][R11] | [19.75][R26] | +3.17 | -0.99 |
| Spatial457 L4 pose | 4555 | [26.37][R31] | [26.81][R12] | [25.99][R27] | +0.44 | -0.37 |
| Spatial457 L5 6D spatial | 1837 | [21.28][R31] | [22.86][R13] | [19.27][R28] | +1.58 | -2.01 |
| Spatial457 L5 collision | 3166 | [23.94][R31] | [24.57][R14] | [24.16][R29] | +0.63 | +0.22 |
| ZeroBench | 100 | [0.00][R31] | [0.00][R15] | [0.00][R30] | +0.00 | +0.00 |

## 补充指标

下表为 ChartQA 和 MMStar 子指标，均为百分数。样本数为所属任务的总问题数，原始 JSON 未单独提供各子类样本数。

| 任务 / 指标 | 样本数 | SigLIP 基线 | ReasonSigLIP S1 | ReasonSigLIP S2 |
| --- | ---: | ---: | ---: | ---: |
| ChartQA human split | 2500 | [54.16][R34] | [55.76][R33] | [54.48][R32] |
| ChartQA augmented split | 2500 | [80.40][R34] | [78.80][R33] | [79.84][R32] |
| MMStar coarse perception | 1500 | [75.37][R34] | [73.54][R33] | [73.54][R32] |
| MMStar fine-grained perception | 1500 | [56.53][R34] | [55.55][R33] | [57.47][R32] |
| MMStar instance reasoning | 1500 | [60.40][R34] | [61.86][R33] | [65.87][R32] |
| MMStar logical reasoning | 1500 | [51.91][R34] | [50.72][R33] | [52.49][R32] |
| MMStar math | 1500 | [50.10][R34] | [48.75][R33] | [51.25][R32] |
| MMStar science & technology | 1500 | [42.32][R34] | [40.15][R33] | [37.78][R32] |

### CoreCognition 按概念准确率

单位为百分数；概念子集样本数未在聚合 JSON 中提供。

| 概念 | SigLIP 基线 | ReasonSigLIP S1 | ReasonSigLIP S2 |
| --- | ---: | ---: | ---: |
| boundary | [75.28][R31] | [79.78][R05] | [69.66][R20] |
| perceptualconstancy | [62.21][R31] | [68.02][R05] | [66.28][R20] |
| conservation | [48.65][R31] | [44.14][R05] | [43.24][R20] |
| toolusing | [88.62][R31] | [86.99][R05] | [85.37][R20] |
| mechanicalreasoning | [68.42][R31] | [74.74][R05] | [72.63][R20] |
| perspective | [29.73][R31] | [28.65][R05] | [31.35][R20] |
| intentionality | [79.28][R31] | [76.58][R05] | [80.18][R20] |
| spatiality | [38.58][R31] | [44.88][R05] | [40.16][R20] |
| intuitivephysics | [53.64][R31] | [54.55][R05] | [54.55][R20] |
| hierarchy | [55.34][R31] | [57.28][R05] | [58.25][R20] |
| permanence | [33.64][R31] | [31.82][R05] | [34.55][R20] |
| continuity | [63.22][R31] | [62.07][R05] | [63.22][R20] |

## 评测设置

- 所有记录的 model 为 `llava_onevision1_5`，`batch_size=1`，`limit=null`，各任务 `n-shot=0`；`n-samples.effective` 与 `original` 一致，未使用 limit 截取子集。
- 模型参数均指定 `attn_implementation=sdpa,max_pixels=3240000`，模型代码路径为 `LLaVA-OneVision-1.5/ds`。
- 同一任务在三个模型间的 `task_hashes`、任务版本、样本数、few-shot 数和任务配置一致；比较配置时去除了 Python 函数表示中的运行时内存地址。
- lmms-eval 版本包含 `dev@7c47168`、`dev@e2061e8`、`dev@a05a9e3`，各结果对应版本见来源表。
- 统计范围为各任务的单次运行结果；按任务分别报告聚合指标及模型间差值。

推理任务的 `max_new_tokens` 来自结果 JSON 的任务配置，其余解码参数保留在原始 JSON 中。

| 任务 | max_new_tokens |
| --- | ---: |
| VisualPuzzles direct | 4096 |
| BLINK IQ test | 1024 |
| BLINK jigsaw | 1024 |
| BLINK multi-view reasoning | 1024 |
| CoreCognition overall | 16 |
| CV-Bench | 1024 |
| MathVision testmini | 16384 |
| Spatial457 L1 single | 512 |
| Spatial457 L2 objects | 512 |
| Spatial457 L3 2D spatial | 512 |
| Spatial457 L4 occlusion | 512 |
| Spatial457 L4 pose | 512 |
| Spatial457 L5 6D spatial | 512 |
| Spatial457 L5 collision | 512 |
| ZeroBench | 128 |

## 已清理权重目录的含义

用户确认三个最终模型已上传 Hugging Face 后，于 2026-10-25 清理了下列六个本地权重目录，共约 96.66 GiB。此处保留目录与训练阶段的对应关系；评测原始文件和本总结继续保留。

路径均相对于原 `outputs/llava_ov15/`，每个目录清理前约 16.11 GiB，包含四个 `.safetensors` 权重分片。

| 目录 | 内容 |
| --- | --- |
| `rs1-1/qwen3_8b_stage0` | 将 Qwen3-8B 与 ReasonSigLIP S1 视觉编码器及连接模块组装得到的初始化模型。 |
| `rs1-1/stage1_alignment` | 558K 图文对齐训练后的模型；冻结视觉编码器和语言模型，只训练连接模块。 |
| `rs1-1/stage1_5_midtraining` | 3M 中期训练后的模型；冻结视觉编码器，训练语言模型和连接模块。 |
| `rs1-1/stage2_instruct` | 使用 LLaVA-NeXT 780K 指令数据训练后的 S1 最终模型，是本次评测对象。 |
| `siglip1/stage2_instruct` | 使用原始 Google SigLIP 视觉编码器的最终基线模型，是本次评测对象。 |
| `reasonsiglip/stage2_instruct` | 使用 ReasonSigLIP S2 视觉编码器的最终模型，是本次评测对象。 |

阶段用途依据 `scripts/llava_ov15_stage0_rs1-1_madar.sh`、`llava_ov15_stage1_rs1-1_madar.sh`、`llava_ov15_stage15_rs1-1_madar.sh`、`llava_ov15_stage2_rs1-1_madar.sh` 及清理前读取的模型配置。每个阶段保存完整模型，因此只训练部分模块的阶段也会占用约 16 GiB。

## 原始文件的用途

| 文件 | 内容 |
| --- | --- |
| `*_results.json` | 聚合指标、标准误、样本数、任务配置和版本，是本总结的数据来源。 |
| `*_samples_*.jsonl` | 逐样本回答与评分，用于检查具体错误和重新分析。 |
| `responses.db` | SQLite 回答缓存，评测可复用已生成回答。 |
| `responses.audit.jsonl` | 缓存写入的 JSONL 日志，可用于恢复缓存。 |
| `submissions/*.json` | 导出的答案提交文件。 |
| `.git/objects/pack/*.pack` | Git 提交历史的压缩对象包。 |

本总结保留聚合指标及来源索引；逐样本回答、标准误和完整运行配置仍以原始 JSON / JSONL 为准。

## 原始结果索引

路径相对于仓库根目录；日期使用结果文件记录的时间字符串，不推断其时区。SHA-256 用于核对来源文件。

| 编号 | 模型 | 评测类别 | 记录时间 | lmms-eval 版本 | 原始结果 | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:48:51 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-1/VisualPuzzles_direct/rs1-1__stage2_instruct/20260807_004851_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/VisualPuzzles_direct/rs1-1__stage2_instruct/20260807_004851_results.json) | `e246953dda24ea4fd24001578efadaeff35d4afc924c6726ad7640d734c1d5a4` |
| R02 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:51:08 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-1/blink_iq_test/rs1-1__stage2_instruct/20260807_005108_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_iq_test/rs1-1__stage2_instruct/20260807_005108_results.json) | `034736c4e5e2b879ce27a6dabf1ae90912ed6ec94bb3b823ea615f5b396501a9` |
| R03 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:51:45 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-1/blink_jigsaw/rs1-1__stage2_instruct/20260807_005145_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_jigsaw/rs1-1__stage2_instruct/20260807_005145_results.json) | `4c4e3e962f720f755171be8b822be8cf38aa5207e7ca131d8a9c957c67195bec` |
| R04 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:52:30 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-1/blink_multi_view_reasoning/rs1-1__stage2_instruct/20260807_005230_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_multi_view_reasoning/rs1-1__stage2_instruct/20260807_005230_results.json) | `ddf8ea069e9fe6a7711144cf3a93e035383cc6050e258b069df3d77893375361` |
| R05 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:53:10 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-1/corecognition/rs1-1__stage2_instruct/20260807_005310_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/corecognition/rs1-1__stage2_instruct/20260807_005310_results.json) | `ad3289d436755b20546cfbae65ecf2bf2e6f30fe8a66a7456d604f1a770df935` |
| R06 | ReasonSigLIP S1 | 推理 | 2026-08-07 00:56:48 | `dev@e2061e8` | [eval_results/llava_ov15/reasoning_suite/rs1-1/cv_bench/rs1-1__stage2_instruct/20260807_005648_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/cv_bench/rs1-1__stage2_instruct/20260807_005648_results.json) | `021b76c737dc53a4f1b26211886df1fa821850b89ceee2c49de05188b5149cd1` |
| R07 | ReasonSigLIP S1 | 推理 | 2026-08-07 01:00:45 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/mathvision_testmini/rs1-1__stage2_instruct/20260807_010045_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/mathvision_testmini/rs1-1__stage2_instruct/20260807_010045_results.json) | `93a0d57b2adda06409185133f87efda118a118db12bf2b3242f73bdd1dd3673b` |
| R08 | ReasonSigLIP S1 | 推理 | 2026-08-07 01:40:56 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l1_single/rs1-1__stage2_instruct/20260807_014056_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l1_single/rs1-1__stage2_instruct/20260807_014056_results.json) | `a168eda824b42dc26bf03f3358996e57b1d5f9dda0dac7bbc026b3ff4f04770b` |
| R09 | ReasonSigLIP S1 | 推理 | 2026-08-07 01:53:00 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l2_objects/rs1-1__stage2_instruct/20260807_015300_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l2_objects/rs1-1__stage2_instruct/20260807_015300_results.json) | `fcef41ef22a4be6f9d1348b5ff5b1b76ac700956ac4aeb77ddf176ff17598084` |
| R10 | ReasonSigLIP S1 | 推理 | 2026-08-07 03:58:24 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l3_2d_spatial/rs1-1__stage2_instruct/20260807_035824_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l3_2d_spatial/rs1-1__stage2_instruct/20260807_035824_results.json) | `d8e1e46dfb4809c2c1f8642657c7292e744761bad147a3d273ccd3c1e04a5045` |
| R11 | ReasonSigLIP S1 | 推理 | 2026-08-07 07:03:07 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_occ/rs1-1__stage2_instruct/20260807_070307_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_occ/rs1-1__stage2_instruct/20260807_070307_results.json) | `b8acae6c4c26c57e451547f222fe61751a25b37806a80e5f41e2aed26e1a514a` |
| R12 | ReasonSigLIP S1 | 推理 | 2026-08-07 09:19:22 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_pose/rs1-1__stage2_instruct/20260807_091922_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_pose/rs1-1__stage2_instruct/20260807_091922_results.json) | `547b9dc0eab0d8e88f7ea38b5fefb3b9af3c4249bbd62ca36b2ceab909fb99ca` |
| R13 | ReasonSigLIP S1 | 推理 | 2026-08-07 10:41:01 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_6d_spatial/rs1-1__stage2_instruct/20260807_104101_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_6d_spatial/rs1-1__stage2_instruct/20260807_104101_results.json) | `4c23f1d43b6f0ea62099b8600ff4ed7a351733677b645492d9dcc073677b71a4` |
| R14 | ReasonSigLIP S1 | 推理 | 2026-08-07 12:10:25 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_collision/rs1-1__stage2_instruct/20260807_121025_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_collision/rs1-1__stage2_instruct/20260807_121025_results.json) | `65006ea960720220643039c209bb9c639f23995078b8d8993949752f91cef0e4` |
| R15 | ReasonSigLIP S1 | 推理 | 2026-08-07 01:39:27 | `dev@a05a9e3` | [eval_results/llava_ov15/reasoning_suite/rs1-1/zerobench/rs1-1__stage2_instruct/20260807_013927_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-1/zerobench/rs1-1__stage2_instruct/20260807_013927_results.json) | `3e0e37a502feae710fb998b1584e7efecb8d1554f09279bd51a33998aa396edd` |
| R16 | ReasonSigLIP S2 | 推理 | 2026-08-03 00:57:31 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/VisualPuzzles_direct/reasonsiglip__stage2_instruct/20260803_005731_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/VisualPuzzles_direct/reasonsiglip__stage2_instruct/20260803_005731_results.json) | `5bde549c493ad91285a2d22f934a1ddacb7ac57296881854bda16fe712fbc4a8` |
| R17 | ReasonSigLIP S2 | 推理 | 2026-08-03 00:59:42 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/blink_iq_test/reasonsiglip__stage2_instruct/20260803_005942_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_iq_test/reasonsiglip__stage2_instruct/20260803_005942_results.json) | `a0049501d9532a37ae8d99b4c4d22f13692ac732d9b398b0ac693e95aef22d7e` |
| R18 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:00:21 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/blink_jigsaw/reasonsiglip__stage2_instruct/20260803_010021_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_jigsaw/reasonsiglip__stage2_instruct/20260803_010021_results.json) | `f61abac8a705eea5363ce977c3b7f19baf95ea2626f2232210dcf8645c8f03cd` |
| R19 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:01:07 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/blink_multi_view_reasoning/reasonsiglip__stage2_instruct/20260803_010107_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_multi_view_reasoning/reasonsiglip__stage2_instruct/20260803_010107_results.json) | `53de50df718cacf3a5413cb89b286d28cadf37bd61c55fc6b26521b653da4541` |
| R20 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:01:43 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/corecognition/reasonsiglip__stage2_instruct/20260803_010143_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/corecognition/reasonsiglip__stage2_instruct/20260803_010143_results.json) | `ab1006974f6cadc60dfd68c5c1a94445f4af22bb558a14fd2b7deeb1ac2c4c80` |
| R21 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:05:17 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/cv_bench/reasonsiglip__stage2_instruct/20260803_010517_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/cv_bench/reasonsiglip__stage2_instruct/20260803_010517_results.json) | `5740fb0b16816f1259b244df01635b90c57ec7d7963b86aa029e7521223ae14d` |
| R22 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:09:24 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/mathvision_testmini/reasonsiglip__stage2_instruct/20260803_010924_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/mathvision_testmini/reasonsiglip__stage2_instruct/20260803_010924_results.json) | `74f96bee9ae20ef84d5182cc0a1332fdc0f815052ca05ed559c50856b75f1d62` |
| R23 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:39:37 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l1_single/reasonsiglip__stage2_instruct/20260803_013937_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l1_single/reasonsiglip__stage2_instruct/20260803_013937_results.json) | `5ac8f6d440746807126bd1ce4830f64f4aabc09bf9e0911ee2a002b4cfda9c6b` |
| R24 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:54:54 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l2_objects/reasonsiglip__stage2_instruct/20260803_015454_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l2_objects/reasonsiglip__stage2_instruct/20260803_015454_results.json) | `b3161551730dc7e95d0ec06516dba5397611f524eb9b2dc57ce59d665cb959c7` |
| R25 | ReasonSigLIP S2 | 推理 | 2026-08-03 04:43:23 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l3_2d_spatial/reasonsiglip__stage2_instruct/20260803_044323_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l3_2d_spatial/reasonsiglip__stage2_instruct/20260803_044323_results.json) | `f6aaf4a6b3b6c085e464d1ea6c4227e412ed72fa0d7c3bec9835b193455ea9d5` |
| R26 | ReasonSigLIP S2 | 推理 | 2026-08-03 08:33:42 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_occ/reasonsiglip__stage2_instruct/20260803_083342_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_occ/reasonsiglip__stage2_instruct/20260803_083342_results.json) | `e43bcc191b30e8c738fb6ea52376e79e1a4eeb79121ff8ffed3cce3cab94e111` |
| R27 | ReasonSigLIP S2 | 推理 | 2026-08-03 11:49:48 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_pose/reasonsiglip__stage2_instruct/20260803_114948_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_pose/reasonsiglip__stage2_instruct/20260803_114948_results.json) | `fd9d3cc6f35a5854aaa89211b6c84e880f84b4de96655a465d376f9615d454f9` |
| R28 | ReasonSigLIP S2 | 推理 | 2026-08-03 13:25:04 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_6d_spatial/reasonsiglip__stage2_instruct/20260803_132504_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_6d_spatial/reasonsiglip__stage2_instruct/20260803_132504_results.json) | `81bc2671194e2566bf44e2f79643407e3cbfa61e948fa3659f47d3fe8778936f` |
| R29 | ReasonSigLIP S2 | 推理 | 2026-08-03 15:17:08 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_collision/reasonsiglip__stage2_instruct/20260803_151708_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_collision/reasonsiglip__stage2_instruct/20260803_151708_results.json) | `6e0fcafff4045fb31ebc4b3b5dbb4ca37a757ddedd7443994917e79bda491c3f` |
| R30 | ReasonSigLIP S2 | 推理 | 2026-08-03 01:38:04 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/rs1-2/zerobench/reasonsiglip__stage2_instruct/20260803_013804_results.json](../eval_results/llava_ov15/reasoning_suite/rs1-2/zerobench/reasonsiglip__stage2_instruct/20260803_013804_results.json) | `790acfb0a0846585a5b18cd238e9b3a9bd3a62b8c0b27635b502c8c2558d3ce4` |
| R31 | SigLIP 基线 | 推理 | 2026-07-31 02:12:01 | `dev@7c47168` | [eval_results/llava_ov15/reasoning_suite/siglip1/siglip1__stage2_instruct/20260731_021201_results.json](../eval_results/llava_ov15/reasoning_suite/siglip1/siglip1__stage2_instruct/20260731_021201_results.json) | `b63e52fbac59cc9fd11bd4f211d27390c8fb34c051d50542ed2322c49a5288f5` |
| R32 | ReasonSigLIP S2 | 通用 | 2026-07-30 19:33:50 | `dev@7c47168` | [eval_results/llava_ov15/reasonsiglip_full/reasonsiglip__stage2_instruct/20260730_193350_results.json](../eval_results/llava_ov15/reasonsiglip_full/reasonsiglip__stage2_instruct/20260730_193350_results.json) | `5a4dc74a1efb8b432c51a1907f5a0a41e08d5254fe19ca0169d14cf767a60d59` |
| R33 | ReasonSigLIP S1 | 通用 | 2026-08-07 00:48:51 | `dev@a05a9e3` | [eval_results/llava_ov15/rs1-1_full/rs1-1__stage2_instruct/20260807_004851_results.json](../eval_results/llava_ov15/rs1-1_full/rs1-1__stage2_instruct/20260807_004851_results.json) | `0ed940f177597a3a4ef04f8f352e5088bc6bf70d3af5f7b556df29b365ad1acf` |
| R34 | SigLIP 基线 | 通用 | 2026-07-30 02:35:35 | `dev@7c47168` | [eval_results/llava_ov15/siglip1_full/siglip1__stage2_instruct/20260730_023535_results.json](../eval_results/llava_ov15/siglip1_full/siglip1__stage2_instruct/20260730_023535_results.json) | `cea93c7d95e096aa69e7f24bcc602b89e5120ec365bc94d44121aedc9ced07f7` |

[R01]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/VisualPuzzles_direct/rs1-1__stage2_instruct/20260807_004851_results.json
[R02]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_iq_test/rs1-1__stage2_instruct/20260807_005108_results.json
[R03]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_jigsaw/rs1-1__stage2_instruct/20260807_005145_results.json
[R04]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/blink_multi_view_reasoning/rs1-1__stage2_instruct/20260807_005230_results.json
[R05]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/corecognition/rs1-1__stage2_instruct/20260807_005310_results.json
[R06]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/cv_bench/rs1-1__stage2_instruct/20260807_005648_results.json
[R07]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/mathvision_testmini/rs1-1__stage2_instruct/20260807_010045_results.json
[R08]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l1_single/rs1-1__stage2_instruct/20260807_014056_results.json
[R09]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l2_objects/rs1-1__stage2_instruct/20260807_015300_results.json
[R10]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l3_2d_spatial/rs1-1__stage2_instruct/20260807_035824_results.json
[R11]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_occ/rs1-1__stage2_instruct/20260807_070307_results.json
[R12]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l4_pose/rs1-1__stage2_instruct/20260807_091922_results.json
[R13]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_6d_spatial/rs1-1__stage2_instruct/20260807_104101_results.json
[R14]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/spatial457_l5_collision/rs1-1__stage2_instruct/20260807_121025_results.json
[R15]: ../eval_results/llava_ov15/reasoning_suite/rs1-1/zerobench/rs1-1__stage2_instruct/20260807_013927_results.json
[R16]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/VisualPuzzles_direct/reasonsiglip__stage2_instruct/20260803_005731_results.json
[R17]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_iq_test/reasonsiglip__stage2_instruct/20260803_005942_results.json
[R18]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_jigsaw/reasonsiglip__stage2_instruct/20260803_010021_results.json
[R19]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/blink_multi_view_reasoning/reasonsiglip__stage2_instruct/20260803_010107_results.json
[R20]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/corecognition/reasonsiglip__stage2_instruct/20260803_010143_results.json
[R21]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/cv_bench/reasonsiglip__stage2_instruct/20260803_010517_results.json
[R22]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/mathvision_testmini/reasonsiglip__stage2_instruct/20260803_010924_results.json
[R23]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l1_single/reasonsiglip__stage2_instruct/20260803_013937_results.json
[R24]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l2_objects/reasonsiglip__stage2_instruct/20260803_015454_results.json
[R25]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l3_2d_spatial/reasonsiglip__stage2_instruct/20260803_044323_results.json
[R26]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_occ/reasonsiglip__stage2_instruct/20260803_083342_results.json
[R27]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l4_pose/reasonsiglip__stage2_instruct/20260803_114948_results.json
[R28]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_6d_spatial/reasonsiglip__stage2_instruct/20260803_132504_results.json
[R29]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/spatial457_l5_collision/reasonsiglip__stage2_instruct/20260803_151708_results.json
[R30]: ../eval_results/llava_ov15/reasoning_suite/rs1-2/zerobench/reasonsiglip__stage2_instruct/20260803_013804_results.json
[R31]: ../eval_results/llava_ov15/reasoning_suite/siglip1/siglip1__stage2_instruct/20260731_021201_results.json
[R32]: ../eval_results/llava_ov15/reasonsiglip_full/reasonsiglip__stage2_instruct/20260730_193350_results.json
[R33]: ../eval_results/llava_ov15/rs1-1_full/rs1-1__stage2_instruct/20260807_004851_results.json
[R34]: ../eval_results/llava_ov15/siglip1_full/siglip1__stage2_instruct/20260730_023535_results.json
