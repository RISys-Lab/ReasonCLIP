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
    def _bce_logits_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        SigLIP 风格的 logistic 对比损失：
        - logits: [B, N]，每一行是一个 query 与所有候选的相似度
        - labels: [B]，每一行中正样本的索引
        """
        targets = torch.zeros_like(logits, dtype=logits.dtype)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)
        # 简单设置正样本权重为 (#neg)
        pos_weight = torch.tensor(logits.shape[1] - 1, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        使用全局负样本（跨 GPU all_gather）的 SigLIP 对比损失，实现真正的大 batch 训练。
        """
        # 1) 前向：拿到图文特征（SigLIP2Model 会返回 image_embeds / text_embeds）
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            return_dict=True,
        )
        image_features = outputs.image_embeds   # [B, D]
        text_features  = outputs.text_embeds    # [B, D]

        # 2) 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features, dim=-1)

        # 3) 定义跨卡 all_gather，保留本地梯度
        def all_gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
            if not (dist.is_available() and dist.is_initialized()):
                return x
            world = dist.get_world_size()
            rank = dist.get_rank()
            xs = [torch.zeros_like(x) for _ in range(world)]
            # 通信时不需要梯度
            dist.all_gather(xs, x.detach())
            # 当前 rank 保留带梯度的本地张量
            xs[rank] = x
            return torch.cat(xs, dim=0)

        B = image_features.size(0)
        device = image_features.device

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # 4) 聚合得到全局特征
        all_image = all_gather_with_local_grad(image_features)  # [world*B, D]
        all_text  = all_gather_with_local_grad(text_features)   # [world*B, D]

        # 全局标签：按照 rank 顺序平移
        labels = torch.arange(B, device=device) + rank * B

        # SigLIP2 有可能带有温度参数，这里如果存在就用上
        logit_scale = getattr(model, "logit_scale", None)
        if logit_scale is not None:
            logit_scale = logit_scale.exp()
        else:
            logit_scale = 1.0

        # 5) 构造对比 logits（每个样本与全局所有样本做对比）
        logits_per_image = logit_scale * (image_features @ all_text.t())   # [B, world*B]
        logits_per_text  = logit_scale * (text_features  @ all_image.t())  # [B, world*B]

        # 6) SigLIP 风格的 logistic 对比损失（图->文、文->图 双向）
        loss_i = self._bce_logits_loss(logits_per_image, labels)
        loss_t = self._bce_logits_loss(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        if return_outputs:
            # 可选：把新的 logits 挂到 outputs 上，方便调试/可视化
            outputs.logits_per_image = logits_per_image.detach()
            outputs.logits_per_text = logits_per_text.detach()
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
    def __init__(self, dataset_dict, processor):
        self.dataset = dataset_dict
        self.processor = processor
    
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
        
        # 使用CLIP处理器处理图像和文本
        encoding = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
            # return_attention_mask=True,  
        )
        
        # 移除批次维度
        batch = {k: v.squeeze(0) for k, v in encoding.items()}
        return batch


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    
    # 创建带时间戳的运行名称
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    args.run_name = f"{args.run_name}_{timestamp}"
    args.output_dir = f"{args.output_dir}_{timestamp}"
    args.best_model_dir = f"{args.best_model_dir}_{timestamp}"
    if args.wandb_log and is_main_process:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name  # 使用带时间戳的名称
        )
    else: os.environ["WANDB_DISABLED"] = "true"
    
    model_name = args.model_name
    # 使用 Siglip2 模型与处理器进行微调，并启用 flash-attn2（环境需已安装 flash-attn 且 GPU 支持）
    # 根据 bf16 / fp16 选定权重精度
    model = Siglip2Model.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )
    processor = Siglip2Processor.from_pretrained(model_name)
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

    print(f"Dataset structure: {raw_ds}")
    print(f"Column names: {raw_ds['train'].column_names}")
    raw_ds = raw_ds['train']
    # take 1000 for demo test
    # raw_ds = raw_ds.shuffle(seed=42).select(range(1000))  # 仅用于演示测试
    
    # 如果数据集没有预定义分割，则手动分割
    split_ds = raw_ds.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]
        
    # 包装为CLIP可用的数据集
    train_dataset = UniFireDataset(train_dataset, processor)
    eval_dataset = UniFireDataset(eval_dataset, processor)


    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16 and (not args.bf16),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if is_main_process else 999999,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,  # 保留更多检查点
        report_to="wandb" if args.wandb_log and is_main_process else "none",
        run_name=args.run_name,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",  # 使用验证损失作为指标
        # greater_is_better=False,       # 损失越小越好
        dataloader_num_workers=args.num_workers,           # 默认: 0 (主进程加载)
        dataloader_pin_memory=True,         # 默认: True
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

    # 过去这里会单独保存“最优模型”到 best_model_dir。
    # 现在改为：只保存最终模型到 output_dir（由 TrainingArguments 控制），并返回该路径。
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
    # 检查当前进程的rank
    import os
    import torch.distributed as dist
    
    # 只有rank0进程执行推送操作
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        # 自动上传 SigLIP2 最终模型到 hub
        model = Siglip2Model.from_pretrained(model_path)
        processor = Siglip2Processor.from_pretrained(model_path)

        model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)

        print(f"Model pushed to HuggingFace Hub: https://huggingface.co/{repo_name}")
    else:
        print(f"Skipping model push for rank {dist.get_rank()}")

    # Ensure all processes synchronize
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    final_model_path = train_clip(args)
    
    if args.push_to_hub and args.hub_username:
        repo_name = f"{args.hub_username}/{args.hub_model_name}"
        push_to_hub(final_model_path, repo_name)


    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print("All processes have completed successfully.")
