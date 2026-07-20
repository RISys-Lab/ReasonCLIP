# 当前已确认的评测协议

只记录已经写入代码的任务；未列出的任务仍未确认。

## 共用设置

| 项目 | 设置 |
| --- | --- |
| CLIP baseline | `openai/clip-vit-large-patch14` |
| SigLIP 1 baseline | `google/siglip-so400m-patch14-384` |
| SigLIP 2 baseline | `google/siglip2-so400m-patch14-384` |
| Backbone | 三者均全程冻结，BF16 |
| Head 参数 | FP32，输入为 patch feature + global feature；CLIP 用 CLS，SigLIP 1/2 都用 MAP |
| Seed | 42 |

## VOC 2012 / ADE20K 语义分割

| 数据集 | 训练集 | 验证集 | 类别 | 论文 sanity mIoU：CLIP / SigLIP 1 / SigLIP 2 |
| --- | ---: | ---: | ---: | ---: |
| VOC 2012 + SBD | 1,464 + 9,118 | 1,449 | 21 | 74.5 / 73.8 / 78.1 |
| ADE20K | 20,210 | 2,000 | 150 | 39.0 / 40.8 / 45.4 |

- 输入：随机缩放 `0.5-2.0`，随机裁剪 `512x512`，水平翻转和颜色增强。
- Head：`BatchNorm2d + 1x1 Conv`，输入通道自动取 `2 * hidden_size`；So/14 的 SigLIP 1/2 路径相同。
- 训练：AdamW，40k steps，batch 16，LR `1e-3`，weight decay `1e-4`，1500-step warmup + poly decay。
- 测试：短边 512，`512x512` 滑窗，stride `341x341`，指标为 mIoU。
- 依据：TIPS v1 和 DINOv2 公开的 VOC/ADE20K linear-head 配置。
- 比较 Reason checkpoint 时保持所有参数不变，只替换 `--model-id`；模型目录需包含其标准 `preprocessor_config.json`，segmentation 入口不接受独立 processor 覆盖。
- 公平结论以 baseline 和 Reason 各自输出 JSON 中的 `metrics.miou` 为准；论文数值只作为 sanity check。

## NYUv2 单目深度

| 项目 | 设置 |
| --- | --- |
| 数据 | BTS synchronized split：24,231 train / 654 test，深度由毫米除以 1000 转为米 |
| 分辨率 | 训练和测试输入均为 `480x640`；训练不做 NYU crop，也不做 `416x544` random crop |
| 训练增强 | 随机旋转 `+/-2.5` 度、水平翻转、gamma/亮度/颜色增强 |
| Patch 对齐 | patch-14 中心补边到 `490x644`，输出再中心裁回 `480x640` |
| Head | 无 BN；`1x1 Conv -> 256` 个等距深度 bin，4x 上采样，范围 `0.001-10 m` |
| 训练 | AdamW，50k steps，batch 8，LR `1e-4`，weight decay `0.01`，12,800-step warmup + cosine decay |
| Loss | SigLoss `1.0` + GradientLoss `0.5` |
| 测试 | Eigen crop + 水平翻转平均；主指标 RMSE，越低越好 |
| 论文 RMSE | CLIP 0.553；SigLIP 1 0.563；SigLIP 2 0.466 |

分辨率和 head 按 TIPS v1；TIPS 未给出的优化器、loss 和增强细节沿用 DINOv2 公开实现。

## NAVI 单目相对深度

| 项目 | 设置 |
| --- | --- |
| 数据 | NAVI v1；有 `wild_set` 的 34 个物体，`multiview_*` 训练、`wild_set` 测试，每 4 帧取 1 帧 |
| 本地样本 | 2,024 trainval / 555 test（stride 后） |
| 输入 | Probe3D 物体方框裁剪后 resize 为 `512x512`；训练使用颜色增强 |
| 深度 | disparity 转米后，逐图归一化为相对深度；无效像素为 0，有效范围 `[0.01, 1.0]` |
| Head | TIPS 官方 4-layer DPT，patch feature 与 global feature 用 project readout 融合；256 个等距 bins，范围 `0.001-1.0` |
| 训练 | AdamW，50k steps，batch 8，LR `5e-4`，weight decay `0.01`，7,500-step warmup + cosine decay |
| Loss | Probe3D SigLoss `10.0` + GradientLoss `0.5` |
| 测试 | 不做 flip/scale-shift；逐图计算 RMSE 后取平均，越低越好 |
| 论文 RMSE | CLIP 0.073；SigLIP 1 0.069；SigLIP 2 0.064 |

NAVI 的 split、相对深度、loss 和 optimizer 沿用 Probe3D；DPT 结构及 batch 8 / 50k steps 按 TIPS。运行时不接受独立 `--processor-id`，比较 Reason checkpoint 时只替换 `--model-id`。

## NYUv2 表面法线

| 项目 | 设置 |
| --- | --- |
| 数据 | Probe3D/GeoNet 的 30,914 个 trainval 样本；NYUv2 labeled test 的 654 个 Ladicky 法线标注 |
| 分辨率 | 训练先从 `480x640` 中心裁成 `480x480`；测试保留完整 `480x640` |
| 训练增强 | ColorJitter `p=0.8`；RandomResizedCrop `p=0.5`、scale `0.5-1.0`；不旋转、不水平翻转 |
| Head | TIPS 官方 4-layer DPT，四个均匀 transformer blocks，project readout 融合 global 与 patch feature，输出 3 通道 |
| 法线输出 | 低分辨率输出先做 L2 normalize，bicubic 上采样到标签尺寸，再做 L2 normalize |
| 训练 | AdamW，50k steps，batch 8，LR `5e-4`，weight decay `0.01`，7,500-step warmup + cosine decay |
| Loss | 在 `depth > 0` 像素上计算 mean angular error（弧度）；不使用第 4 个 uncertainty channel |
| 测试 | 每张图计算 angular RMSE（度）后取平均，同时输出 `<11.25°`、`<22.5°`、`<30°` recall |
| 论文 angular RMSE | CLIP 24.3；SigLIP 1 24.1；SigLIP 2 23.0 |

## NAVI 表面法线

| 项目 | 设置 |
| --- | --- |
| 数据 | 与 NAVI-D 相同：34 个有效物体，2,024 trainval / 555 test，`multiview_*` 训练、`wild_set` 测试、stride 4 |
| 输入 | Probe3D object bbox crop 后 resize 到 `512x512`；训练使用颜色增强 |
| 法线标签 | 保留 metric depth，由深度和相机焦距生成 camera-space normals；坐标为 `+x` 向右、`+y` 向下、`+z` 指向相机内部 |
| Head/输出 | 与 NYUv2-N 相同的 TIPS 3-channel 4-layer DPT，以及 normalize → bicubic → normalize |
| 训练 | AdamW，50k steps，batch 8，LR `5e-4`，weight decay `0.01`，7,500-step warmup + cosine decay |
| Loss/测试 | `depth > 0` masked angular loss；逐图 angular RMSE 与三个角度 recall |
| 论文 angular RMSE | CLIP 25.5；SigLIP 1 25.4；SigLIP 2 25.0 |

Normal 的数据和指标沿用 Probe3D，DPT、分辨率、batch 8 和 50k steps 按 TIPS。TIPS 未公开完整 optimizer 配置，因此 LR、warmup、cosine 和 AdamW 细节沿用 Probe3D；3 通道输出及 bicubic 前后归一化按当前 TIPS 官方 decoder/notebook，而不是 Probe3D 的 4 通道 uncertainty head。

## 运行

```bash
python eval/eval_nyuv2_depth.py \
  --model-id openai/clip-vit-large-patch14

python eval/eval_navi_depth.py \
  --model-id google/siglip-so400m-patch14-384

python eval/eval_nyuv2_normals.py \
  --model-id openai/clip-vit-large-patch14

python eval/eval_navi_normals.py \
  --model-id google/siglip-so400m-patch14-384
```

SigLIP 2 或 Reason checkpoint 使用相同命令，只替换 `--model-id`；需要自定义结果文件名时再加 `--model-name`。默认数据根目录是 `data/downstream_data`，结果写入 `eval/results/downstream`。

主要依据：[TIPS 论文](https://arxiv.org/abs/2410.16512)、[TIPS 官方 DPT](https://github.com/google-deepmind/tips/blob/main/pytorch/decoders.py)、[Probe3D](https://github.com/mbanani/probe3d) 和 [SigLIP 2 Table 2](https://arxiv.org/abs/2502.14786)。
