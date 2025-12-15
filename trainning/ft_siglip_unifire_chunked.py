import os
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import (
    Siglip2Processor,
    Siglip2Model,
    SiglipProcessor,
    SiglipModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
from accelerate import Accelerator
import math
import torch.distributed as dist

# 初始化 accelerator
accelerator = Accelerator()

def main_print(*args, **kwargs):
    """只在主进程打印的函数"""
    if accelerator.is_main_process:
        print(*args, **kwargs)

# 非主进程立即禁用 Wandb（在导入wandb之前）
if not accelerator.is_main_process:
    os.environ["WANDB_DISABLED"] = "true"
    print("Wandb disabled for non-main processes")

import wandb
import argparse
from PIL import Image
import io
from transformers import TrainerCallback
from datetime import datetime


MODEL_LOCAL_PATH = "/leonardo_work/EUHPC_R04_192/fmohamma/my_hf_cache/transformers/google/siglip2-so400m-patch14-384"
DATASET_LOCAL_PATH = "/leonardo_work/EUHPC_R04_192/fmohamma/my_hf_cache/datasets/fesvhtr/iferniu"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    # Training parameters
    
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch16-naflex", 
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./weights/unifire_siglip_finetune", 
                        help="Output directory")
    parser.add_argument("--best_model_dir", type=str, default="./weights/unifire_siglip_best_model", 
                        help="Directory to save the best model")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, 
                        help="Learning rate")
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
    parser.add_argument("--logging_steps", type=int, default=25, 
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Steps to save checkpoints")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Evaluation steps")
    parser.add_argument("--run_name", type=str, default="siglip-finetune-unifire", 
                        help="Experiment name")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for gradient clipping")
    # parser.add_argument("--early_stopping_patience", type=int, default=3,
    #                     help="Patience for early stopping")
    
    # Hub push parameters
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="fesvhtr", 
                        help="HuggingFace username")
    parser.add_argument("--hub_model_name", type=str, default="siglip-iferniu-L14-10epoch", 
                        help="Model name on the Hub")
    
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
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of workers for data loading")

                        
    # wandb parameters
    parser.add_argument("--wandb_project", type=str, default="siglip-unifire",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity name (team or username)")
    parser.add_argument("--wandb_log", action="store_true",
                        help="Enable wandb logging")
    
    return parser.parse_args()
    
    
# class BestModelCallback(TrainerCallback):
#     """
#     原先用于根据 eval_loss 追踪和打印“最优模型”的回调。
#     现在不再单独保存/追踪最优模型，因此整体注释掉。
#     如需恢复，只需取消本类以及 Trainer 中 callbacks 的注释。
#     """
#     def __init__(self):
#         self.best_eval_loss = float('inf')
#         
#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         if metrics and "eval_loss" in metrics:
#             eval_loss = metrics["eval_loss"]
#
#             is_main_process = not dist.is_initialized() or dist.get_rank() == 0
#             
#             # 手动记录到 Wandb
#             if is_main_process and (args.report_to == "wandb" or (isinstance(args.report_to, list) and "wandb" in args.report_to)):
#                 import wandb
#                 if wandb.run is not None:
#                     wandb.log({"eval_loss": eval_loss}, step=state.global_step)
#                     print(f"已手动记录 eval_loss={eval_loss:.4f} 到 Wandb (step={state.global_step})")
#             
#             # 检查是否为新的最佳模型
#             if eval_loss < self.best_eval_loss:
#                 print(f"\n>>> eval_loss: {eval_loss:.4f}\n")
#                 self.best_eval_loss = eval_loss
#                 print(f"\n*** New best model: {state.global_step}, Loss: {self.best_eval_loss:.4f} ***\n")

class SiglipTrainer(Trainer):
    @staticmethod
    def _siglip_logistic_loss_full(label_matrix: torch.Tensor,
                                   logits: torch.Tensor) -> torch.Tensor:
        """
        对给定的 (+1 / -1) label_matrix 和 logits 做 SigLIP logistic：
        L = - 1/B * sum_i sum_j log σ(y_ij * s_ij)
        这里不关心正样本的位置，调用方自己构造 label_matrix。
        """
        per_pair = -F.logsigmoid(label_matrix * logits)  # [B, B]
        # 和论文一样：sum over j，再对 batch 做 mean，等价于 sum / B
        loss = per_pair.sum(dim=1).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        按 SigLIP 3.3 节的思路做“chunked / ring”实现：
        - 只保留本卡的 image/text，永远不物化 |B_global|×|B_global| 的大矩阵；
        - 第 0 轮用本卡 text（含正样本，带梯度）；
        - 之后 world_size-1 轮用 ring 方式交换 text（只做负样本，detach()）。
        """

        # ---------- 1. forward：取 image / text features ----------
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
        image_features = outputs.image_embeds   # [B, D]
        text_features  = outputs.text_embeds    # [B, D]

        # normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)

        B = image_features.size(0)
        device = image_features.device
        world_size = accelerator.num_processes
        rank = accelerator.process_index

        # ---------- 2. 取 backbone + clamp logit_scale ----------
        backbone = model.module if hasattr(model, "module") else model

        with torch.no_grad():
            if hasattr(backbone, "logit_scale"):
                backbone.logit_scale.data.clamp_(max=math.log(100.0))

        if hasattr(backbone, "logit_scale"):
            logit_scale = backbone.logit_scale.exp()
        else:
            logit_scale = image_features.new_tensor(1.0)

        logit_bias_param = getattr(backbone, "logit_bias", None)
        if logit_bias_param is not None:
            bias = logit_bias_param.to(image_features.dtype)
        else:
            bias = 0.0

        # ---------- 3. ring-chunk loss：第 0 轮本地块，后面轮负样本块 ----------
        # 第 0 轮：本卡 text（带梯度），既有正样本又有本地负样本
        logits_local = logit_scale * (image_features @ text_features.t()) + bias  # [B, B]

        # 构造本地块的 label_matrix：对角线 +1，其余 -1
        label_matrix_local = logits_local.new_full(logits_local.shape, -1.0)
        idx = torch.arange(B, device=device)
        label_matrix_local[idx, idx] = 1.0

        loss = self._siglip_logistic_loss_full(label_matrix_local, logits_local)

        # 之后 world_size-1 轮：只交换负样本块（全 -1），不需要梯度
        if world_size > 1:
            neg_chunk = text_features.detach()          # 当前持有的负样本块
            origin_rank = rank                          # neg_chunk 来自哪个 rank

            for _ in range(world_size - 1):
                # ring 交换：把当前块发给 next，从 prev 收一块
                next_rank = (rank + 1) % world_size
                prev_rank = (rank - 1 + world_size) % world_size

                send_buf = neg_chunk
                recv_buf = torch.empty_like(neg_chunk)

                dist.send(send_buf, dst=next_rank)
                dist.recv(recv_buf, src=prev_rank)

                neg_chunk = recv_buf
                origin_rank = (origin_rank - 1 + world_size) % world_size

                # 对这块 neg_chunk 计算 logits；所有 pair 都是负样本（label = -1）
                logits_neg = logit_scale * (image_features @ neg_chunk.t()) + bias
                label_matrix_neg = logits_neg.new_full(logits_neg.shape, -1.0)

                loss = loss + self._siglip_logistic_loss_full(
                    label_matrix_neg, logits_neg
                )

        # loss 现在已经是 sum_{all chunks} (1/B * sum_j per_pair_ij)，
        # 等价于论文里的 1/|B_global| * sum_i sum_j L_ij（这里 |B_global| = B * world_size，
        # 但每个样本只在各自 rank 上除以 B；DDP 同步梯度后是等价的）。

        if return_outputs:
            return loss, outputs
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 调用父类的评估方法
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # 确保评估结果中包含 loss 指标
        if f"{metric_key_prefix}_loss" not in metrics:
            # 如果父类没有计算loss，我们手动计算
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            total_loss = 0.0
            num_batches = 0
            
            # 将模型设为评估模式
            self.model.eval()
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    # 准备输入
                    batch = self._prepare_inputs(batch)
                    
                    # 计算损失
                    loss = self.compute_loss(self.model, batch)
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            # 计算平均损失
            if num_batches > 0:
                metrics[f"{metric_key_prefix}_loss"] = total_loss / num_batches
        self.log(metrics)
        return metrics

class UniFireDataset(torch.utils.data.Dataset):  # 修正继承
    def __init__(self, dataset_dict, processor, is_naflex: bool = False):
        self.dataset = dataset_dict
        self.processor = processor
        # 是否为 NaFlex 变体，用于决定是否传 max_num_patches / spatial_shapes 等参数
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
        
        # 使用 SigLIP2 的 processor 处理图像和文本
        # - FixRes: 固定分辨率，不需要 max_num_patches / spatial_shapes
        # - NaFlex: 需要 max_num_patches，processor 会自动生成 spatial_shapes / pixel_attention_mask
        if self.is_naflex:
            encoding = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=64,       # 官方推荐：文本长度固定 64
                truncation=True,
                max_num_patches=256, # 官方 NaFlex 示例：控制最大 patch 数，可按显存调大/调小
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
        
        # 移除批次维度
        batch = {k: v.squeeze(0) for k, v in encoding.items()}
        return batch


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = accelerator.is_main_process

    # 创建统一的 run_id（只由主进程生成一次，其它进程通过 DDP 广播获得）
    if accelerator.is_main_process:
        run_id = datetime.now().strftime("%m%d_%H%M%S")
    else:
        run_id = None

    if accelerator.num_processes > 1:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            obj_list = [run_id]
            dist.broadcast_object_list(obj_list, src=0)
            run_id = obj_list[0]

    # 使用统一的 run_id 构造 run_name 和输出目录，避免每个 rank 各自建一套目录
    args.run_name = f"{args.run_name}_{run_id}"
    args.output_dir = f"{args.output_dir}_{run_id}"
    args.best_model_dir = f"{args.best_model_dir}_{run_id}"
    if args.wandb_log and is_main_process:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name  # 使用带时间戳的名称
        )
    else: 
        os.environ["WANDB_DISABLED"] = "true"
    
    model_name = args.model_name
    # 加载配置以判断是 SigLIP 还是 SigLIP2，以及是否为 NaFlex 变体
    cfg = AutoConfig.from_pretrained(model_name)
    is_naflex = "naflex" in model_name

    attn_impl = "flash_attention_2"

    if cfg.model_type == "siglip2":
        model = Siglip2Model.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        processor = Siglip2Processor.from_pretrained(model_name)
    elif cfg.model_type == "siglip":
        # 兼容旧的 SigLIP checkpoint（非 SigLIP2）
        model = SiglipModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        processor = SiglipProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model_type '{cfg.model_type}' for checkpoint: {model_name}")
    # Download dataset: 优先使用本地 parquet 文件夹，其次使用 Hub 上的数据集名
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
            "parquet",                             # 告诉 datasets 这是一个本地 Parquet 文件集
            data_files={"train": parquet_files},   # 把所有的 parquet 放在 train 里
        )
    elif getattr(args, "dataset_name", None):
        dataset_name = args.dataset_name
        raw_ds = load_dataset(dataset_name)
    else:
        raise ValueError("Please specify either --dataset-path (folder with parquet files) or --dataset_name.")

    main_print(f"Dataset structure: {raw_ds}")
    main_print(f"Column names: {raw_ds['train'].column_names}")
    raw_ds = raw_ds['train']
    # take 1000 for demo test
    # raw_ds = raw_ds.shuffle(seed=42).select(range(1000))  # 仅用于演示测试
    
    # 如果数据集没有预定义分割，则手动分割
    split_ds = raw_ds.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]
        
    # 包装为 SigLIP2 可用的数据集（根据是否 NaFlex 选择不同预处理）
    train_dataset = UniFireDataset(train_dataset, processor, is_naflex=is_naflex)
    eval_dataset = UniFireDataset(eval_dataset, processor, is_naflex=is_naflex)


    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=args.bf16,
        fp16=args.fp16 and (not args.bf16),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,  # 所有进程使用相同的 save_steps，但只有主进程实际写入
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,  # 增加保留的检查点数量，减少删除操作
        save_only_model=False,  # 如果需要恢复训练，设为 False；如果只需要模型权重，设为 True 可加速保存
        # 只在主进程且未显式禁用 wandb 时上报到 wandb，其它 rank 一律 'none'
        report_to="wandb" if args.wandb_log and is_main_process and not os.environ.get("WANDB_DISABLED") else "none",
        run_name=args.run_name,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",  # 使用验证损失作为指标
        # greater_is_better=False,       # 损失越小越好
        dataloader_num_workers=args.num_workers,           # 默认: 0 (主进程加载)
        dataloader_pin_memory=True, 
        remove_unused_columns=False,  # 与ft_clip_r_s1.py一致
        ddp_find_unused_parameters=True,  # 关闭unused parameters检测，提高性能（与ft_clip_r_s1.py一致）
        dataloader_drop_last=True,
        seed=42,  # 与ft_clip_r_s1.py一致
        data_seed=42,  # 与ft_clip_r_s1.py一致
    )
    

    trainer = SiglipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # 不再追踪/打印“最优模型”，如需恢复可把 BestModelCallback 取消注释后加回下面这一行
        # callbacks=[BestModelCallback()]
    )

    trainer.train()

    # 过去这里会单独保存"最优模型"到 best_model_dir。
    # 现在改为：只由主进程保存最终模型到 output_dir，并返回该路径。
    accelerator.wait_for_everyone()  # 等待所有进程完成训练
    
    final_model_path = args.output_dir
    if accelerator.is_main_process:
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        main_print(f"Final model saved to {final_model_path}")
    else:
        # 非主进程不再重复保存，只是复用相同的路径返回值
        pass

    accelerator.wait_for_everyone()  # 确保保存后再继续
    return final_model_path


def push_to_hub(model_path, repo_name):
    # 只有主进程执行推送操作
    if accelerator.is_main_process:
        main_print(f"Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        # 自动上传 SigLIP2 最终模型到 hub
        model = Siglip2Model.from_pretrained(model_path)
        processor = Siglip2Processor.from_pretrained(model_path)

        model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)

        main_print(f"Model pushed to HuggingFace Hub: https://huggingface.co/{repo_name}")
    
    # 等待所有进程同步
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parse_args()
    final_model_path = train_clip(args)
    
    if args.push_to_hub and args.hub_username:
        repo_name = f"{args.hub_username}/{args.hub_model_name}"
        push_to_hub(final_model_path, repo_name)

    # 等待所有进程完成
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        main_print("All processes have completed successfully.")
