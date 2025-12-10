import os
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import (
    Siglip2Processor,
    Siglip2Model,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
import torch.distributed as dist


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 非主进程立即禁用 Wandb（在导入wandb之前）
if not is_main_process():
    os.environ["WANDB_DISABLED"] = "true"
    print("已禁用非主进程的 Wandb")

import wandb
import argparse
from PIL import Image
import io
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Ring-style SigLIP2 finetune with lower peak memory")
    # Training parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/siglip2-so400m-patch16-naflex",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./weights/unifire_siglip_ring_finetune",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision training",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision training (Ampere+ GPUs)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=25,
        help="Logging steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Steps to save checkpoints",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=250,
        help="Evaluation steps",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="siglip-ring-finetune-unifire",
        help="Experiment name",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping",
    )

    # Hub push parameters
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_username",
        type=str,
        default="fesvhtr",
        help="HuggingFace username",
    )
    parser.add_argument(
        "--hub_model_name",
        type=str,
        default="siglip-iferniu-L14-10epoch-ring",
        help="Model name on the Hub",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fesvhtr/iferniu",
        help="Dataset name on HuggingFace Hub",
    )
    parser.add_argument(
        "--dataset_path",
        "--dataset-path",
        type=str,
        default=None,
        help="本地 parquet 数据集所在文件夹路径（包含若干 *.parquet 文件）",
    )
    default_workers = min(8, os.cpu_count() // 2)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_workers,
        help="Number of workers for data loading",
    )

    # wandb parameters
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="siglip-unifire-ring",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity name (team or username)",
    )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="Enable wandb logging",
    )

    return parser.parse_args()


class RingSiglipTrainer(Trainer):
    @staticmethod
    def _bce_logits_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        SigLIP 风格的 logistic 对比损失：
        - logits: [B, N]，每一行是一个 query 与所有候选的相似度
        - labels: [B]，每一行中正样本的索引
        """
        targets = torch.zeros_like(logits, dtype=logits.dtype)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)
        # 简单设置正样本权重为 (#neg)
        pos_weight = torch.tensor(
            logits.shape[1] - 1,
            device=logits.device,
            dtype=logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        使用 SigLIP logistic loss 的 ring 轮转实现：
        - 不再一次性 all_gather 全局特征，而是固定本卡图像/文本特征，
          通过多轮跨卡“轮转”引入更多负样本，从而降低峰值显存和通信。
        """
        # 0) 获取分布式信息
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 1) 前向：拿到图文特征（SigLIP2Model 会返回 image_embeds / text_embeds）
        #    - FixRes: 只需要 input_ids / pixel_values
        #    - NaFlex: 还会在 inputs 中带上 spatial_shapes / pixel_attention_mask，我们检测到就一并传进去
        model_kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )
        if "spatial_shapes" in inputs:
            model_kwargs["spatial_shapes"] = inputs["spatial_shapes"]
        if "pixel_attention_mask" in inputs:
            model_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]

        outputs = model(**model_kwargs)
        image_features = outputs.image_embeds  # [B, D]
        text_features = outputs.text_embeds  # [B, D]

        # 2) 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        B = image_features.size(0)
        device = image_features.device

        # 3) logit_scale（如果模型带这个参数）
        logit_scale = getattr(model, "logit_scale", None)
        if logit_scale is not None:
            logit_scale = logit_scale.exp()
        else:
            logit_scale = 1.0

        # --------- 情况一：单卡训练，直接用双向 SigLIP loss ----------
        if world_size == 1:
            logits_it = logit_scale * (image_features @ text_features.t())  # [B, B]
            logits_ti = logit_scale * (text_features @ image_features.t())  # [B, B]
            labels = torch.arange(B, device=device)
            loss_i = self._bce_logits_loss(logits_it, labels)
            loss_t = self._bce_logits_loss(logits_ti, labels)
            loss = (loss_i + loss_t) / 2

            if return_outputs:
                outputs.logits_per_image = logits_it.detach()
                outputs.logits_per_text = logits_ti.detach()
                return loss, outputs
            return loss

        # --------- 情况二：多卡 ring 轮转 ----------
        # Step 0：本卡局部块（包含正样本）
        labels_local = torch.arange(B, device=device)

        logits_it_local = logit_scale * (image_features @ text_features.t())
        logits_ti_local = logit_scale * (text_features @ image_features.t())

        loss_i = self._bce_logits_loss(logits_it_local, labels_local)
        loss_t = self._bce_logits_loss(logits_ti_local, labels_local)

        # 只在本地块上对 text/image 都回传梯度
        loss = (loss_i + loss_t) / 2

        # 准备 ring 缓冲（只对负样本部分回传到本卡特征，远端特征使用 detach）
        text_buf = text_features.detach()
        image_buf = image_features.detach()

        # 在 ring 中，我们只追加“全负”块的损失（没有正样本），
        # 这样每张卡的每个样本最终看到 world_size * B - 1 个负样本，
        # 但峰值内存始终只需要一个 B×B logits 和一块 B×D 的远端特征。

        for _ in range(world_size - 1):
            # ---- 文本特征 ring：固定本卡 image_features，引入其它卡 text_buf 作为负样本 ----
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            send_req = dist.isend(text_buf, dst=send_rank)
            recv_buf_t = torch.empty_like(text_buf)
            dist.recv(recv_buf_t, src=recv_rank)
            send_req.wait()
            text_buf = recv_buf_t

            logits_neg_it = logit_scale * (image_features @ text_buf.t())  # [B, B]
            # 全负样本：targets 全 0 即可
            targets_zero_it = torch.zeros_like(logits_neg_it, dtype=logits_neg_it.dtype)
            loss_neg_it = F.binary_cross_entropy_with_logits(
                logits_neg_it,
                targets_zero_it,
            )

            # ---- 图像特征 ring：固定本卡 text_features，引入其它卡 image_buf 作为负样本 ----
            send_req = dist.isend(image_buf, dst=send_rank)
            recv_buf_i = torch.empty_like(image_buf)
            dist.recv(recv_buf_i, src=recv_rank)
            send_req.wait()
            image_buf = recv_buf_i

            logits_neg_ti = logit_scale * (text_features @ image_buf.t())  # [B, B]
            targets_zero_ti = torch.zeros_like(logits_neg_ti, dtype=logits_neg_ti.dtype)
            loss_neg_ti = F.binary_cross_entropy_with_logits(
                logits_neg_ti,
                targets_zero_ti,
            )

            # 把负样本损失累加进来
            loss = loss + (loss_neg_it + loss_neg_ti) / 2

        # 可选：对 loss 做一次 all_reduce，确保多卡取平均
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        if return_outputs:
            # 仅保存本地块 logits，避免额外内存占用
            outputs.logits_per_image = logits_it_local.detach()
            outputs.logits_per_text = logits_ti_local.detach()
            return loss, outputs

        return loss


class UniFireDataset(torch.utils.data.Dataset):
    """
    兼容 FixRes / NaFlex 的 Unifire 数据集封装：
    - NaFlex: 通过 max_num_patches 控制 patch 数，processor 会自动生成 spatial_shapes / pixel_attention_mask
    - FixRes: 固定分辨率，不需要 max_num_patches
    """

    def __init__(self, dataset_dict, processor, is_naflex: bool = False):
        self.dataset = dataset_dict
        self.processor = processor
        self.is_naflex = is_naflex

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 处理bytes格式的图像
        if isinstance(item["image"], dict) and "bytes" in item["image"]:
            image_bytes = item["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(item["image"], bytes):
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
        else:
            image = item["image"]

        label = item["label"]
        caption = item["caption"]
        text = f"A photo of {label}, where {caption}"

        if self.is_naflex:
            encoding = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
                max_num_patches=256,
            )
        else:
            encoding = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
                truncation=True,
            )

        batch = {k: v.squeeze(0) for k, v in encoding.items()}
        return batch


def train_siglip_ring(args):
    # 获取当前是否为主进程
    main_proc = not dist.is_initialized() or dist.get_rank() == 0

    # 创建带时间戳的运行名称
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    args.run_name = f"{args.run_name}_{timestamp}"
    args.output_dir = f"{args.output_dir}_{timestamp}"

    if args.wandb_log and main_proc:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    model_name = args.model_name
    is_naflex = "naflex" in model_name

    # NaFlex 使用 SDPA，FixRes 可以开启 flash-attn2
    attn_impl = "flash_attention_2"
    model = Siglip2Model.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16
        if args.bf16
        else (torch.float16 if args.fp16 else None),
    )
    processor = Siglip2Processor.from_pretrained(model_name)

    # 加载数据集：优先使用本地 parquet 文件夹，其次使用 Hub 数据集名
    if getattr(args, "dataset_path", None):
        parquet_dir = args.dataset_path
        parquet_files = [
            os.path.join(parquet_dir, fname)
            for fname in os.listdir(parquet_dir)
            if fname.endswith(".parquet")
        ]
        if not parquet_files:
            raise ValueError(f"No .parquet files found in folder: {parquet_dir}")
        raw_ds = load_dataset(
            "parquet",
            data_files={"train": parquet_files},
        )
    elif getattr(args, "dataset_name", None):
        raw_ds = load_dataset(args.dataset_name)
    else:
        raise ValueError(
            "Please specify either --dataset-path (folder with parquet files) or --dataset_name."
        )

    print(f"Dataset structure: {raw_ds}")
    print(f"Column names: {raw_ds['train'].column_names}")
    raw_ds = raw_ds["train"]

    split_ds = raw_ds.train_test_split(test_size=0.02, seed=42)
    train_hf = split_ds["train"]
    eval_hf = split_ds["test"]

    train_dataset = UniFireDataset(train_hf, processor, is_naflex=is_naflex)
    eval_dataset = UniFireDataset(eval_hf, processor, is_naflex=is_naflex)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16 and (not args.bf16),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if main_proc else 999999,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        report_to="wandb" if args.wandb_log and main_proc else "none",
        run_name=args.run_name,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    trainer = RingSiglipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    if not dist.is_initialized() or dist.get_rank() == 0:
        final_model_path = args.output_dir
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")
    else:
        final_model_path = args.output_dir
        print(f"Skipping saving final model for rank {dist.get_rank()}")

    return final_model_path


def push_to_hub(model_path, repo_name):
    # 只有 rank0 进程执行推送
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Pushing model to HuggingFace Hub: {repo_name}")
        model = Siglip2Model.from_pretrained(model_path)
        processor = Siglip2Processor.from_pretrained(model_path)
        model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)
        print(f"Model pushed to HuggingFace Hub: https://huggingface.co/{repo_name}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    final_model_path = train_siglip_ring(args)

    if args.push_to_hub and args.hub_username:
        repo_name = f"{args.hub_username}/{args.hub_model_name}"
        push_to_hub(final_model_path, repo_name)

    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print("All processes have completed successfully.")


