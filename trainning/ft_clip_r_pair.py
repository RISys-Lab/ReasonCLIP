import os
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
from accelerate import Accelerator
import numpy as np 
from typing import Optional, List
# import sys
# sys.stderr.isatty = lambda: True
# 初始化 accelerator
accelerator = Accelerator()

def main_print(*args, **kwargs):
    """只在主进程打印的函数"""
    if accelerator.is_main_process:
        print(*args, **kwargs)

# 非主进程立即禁用 Wandb（在导入wandb之前）
if not accelerator.is_main_process:
    os.environ["WANDB_DISABLED"] = "true"
    print("Wandb disabled for non-main processes")  # 这个保留，让每个进程都知道自己的状态

import wandb
import argparse
from transformers import TrainerCallback
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    # Training parameters
    
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", 
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./weights/unifire_clip_finetune", 
                        help="Output directory")
    parser.add_argument("--best_model_dir", type=str, default="./weights/unifire_clip_best_model", 
                        help="Directory to save the best model")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    
    # Logging parameters - support both percentage and steps
    parser.add_argument("--logging_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Logging strategy: 'steps', 'epoch', or 'ratio' (percentage of total steps)")
    parser.add_argument("--logging_steps", type=int, default=25, 
                        help="Logging steps (used when logging_strategy='steps')")
    parser.add_argument("--logging_ratio", type=float, default=0.02,
                        help="Logging ratio (used when logging_strategy='ratio'), e.g., 0.02 = every 2% of total steps")
    
    # Save parameters - support both percentage and steps
    parser.add_argument("--save_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Save strategy: 'steps', 'epoch', or 'ratio' (percentage of total steps)")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Steps to save checkpoints (used when save_strategy='steps')")
    parser.add_argument("--save_ratio", type=float, default=0.1,
                        help="Save ratio (used when save_strategy='ratio'), e.g., 0.1 = every 10% of total steps")
    
    # Evaluation parameters - support both percentage and steps
    parser.add_argument("--eval_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Evaluation strategy: 'steps', 'epoch', or 'ratio' (percentage of total steps)")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Evaluation steps (used when eval_strategy='steps')")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Evaluation ratio (used when eval_strategy='ratio'), e.g., 0.05 = every 5% of total steps")
    
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Total number of checkpoints to save")
    parser.add_argument("--run_name", type=str, default="clip-finetune-unifire", 
                        help="Experiment name")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for gradient clipping")
    # parser.add_argument("--early_stopping_patience", type=int, default=3,
    #                     help="Patience for early stopping")
    
    # Loss weight parameters
    parser.add_argument("--tb_alpha", type=float, default=0.5,
                        help="Weight for tb loss (trp weight = 1 - tb_alpha), range [0, 1]")
    
    # Hub push parameters
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="fesvhtr", 
                        help="HuggingFace username")
    parser.add_argument("--hub_model_name", type=str, default="clip-iferniu-L14-10epoch", 
                        help="Model name on the Hub")
    
    # Dataset parameters
    parser.add_argument("--parquet_file", type=str, required=True,
                        help="Path to the parquet file containing image_path, tb, and trp columns")
    parser.add_argument("--use_split", action="store_true",
                        help="Whether to split dataset into train:eval:test = 8:1:1")
    
    default_workers = min(8, os.cpu_count() // 2)
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of workers for data loading")

    # wandb parameters
    parser.add_argument("--wandb_project", type=str, default="clip-unifire",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity name (team or username)")
    parser.add_argument("--wandb_log", action="store_true",
                        help="Enable wandb logging")
    
    return parser.parse_args()



class BestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float('inf')
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]

            is_main_process = accelerator.is_main_process
            
            # 手动记录到 Wandb
            if is_main_process and (args.report_to == "wandb" or (isinstance(args.report_to, list) and "wandb" in args.report_to)):
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval_loss": eval_loss}, step=state.global_step)
                    main_print(f"Manual wandb logging: eval_loss={eval_loss:.4f} (step={state.global_step})")
            
            # 检查是否为新的最佳模型
            if eval_loss < self.best_eval_loss:
                main_print(f"\n>>> eval_loss: {eval_loss:.4f}\n")
                self.best_eval_loss = eval_loss
                main_print(f"\n*** New best model: {state.global_step}, Loss: {self.best_eval_loss:.4f} ***\n")

class CLIPTrainer(Trainer):
    def __init__(self, tb_alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_weight = tb_alpha
        self.trp_weight = 1.0 - tb_alpha
        # Loss weights already printed in main function
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute CLIP contrastive loss, calculate image-tb and image-trp loss separately
        """

        # 简化方法：将两个损失合并到一次前向传播中
        batch_size = inputs["pixel_values"].size(0)
        device = inputs["pixel_values"].device
        
        # 将TB和TRP的text inputs拼接起来，一次性计算
        combined_input_ids = torch.cat([inputs["tb_input_ids"], inputs["trp_input_ids"]], dim=0)
        combined_attention_mask = torch.cat([inputs["tb_attention_mask"], inputs["trp_attention_mask"]], dim=0)
        combined_pixel_values = torch.cat([inputs["pixel_values"], inputs["pixel_values"]], dim=0)
        
        # 一次前向传播
        combined_outputs = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pixel_values=combined_pixel_values
        )
        
        # 分离TB和TRP的输出
        logits_per_image = combined_outputs.logits_per_image
        logits_per_text = combined_outputs.logits_per_text
        
        # 前半部分是TB，后半部分是TRP
        tb_logits_per_image  = logits_per_image[:batch_size, :batch_size]
        tb_logits_per_text   = logits_per_text [:batch_size, :batch_size]
        trp_logits_per_image = logits_per_image[batch_size:, batch_size:]
        trp_logits_per_text  = logits_per_text [batch_size:, batch_size:]
        
        # 计算损失
        labels = torch.arange(batch_size, device=device)
        
        # TB损失
        tb_loss_i = F.cross_entropy(tb_logits_per_image, labels)
        tb_loss_t = F.cross_entropy(tb_logits_per_text, labels)
        tb_loss = (tb_loss_i + tb_loss_t) / 2.0
        
        # TRP损失  
        trp_loss_i = F.cross_entropy(trp_logits_per_image, labels)
        trp_loss_t = F.cross_entropy(trp_logits_per_text, labels)
        trp_loss = (trp_loss_i + trp_loss_t) / 2.0
        
        # 组合损失
        total_loss = self.tb_weight * tb_loss + self.trp_weight * trp_loss

        if accelerator.is_main_process and "wandb" in self.args.report_to:
            # commit=False，等 Transformer 自己在 step 末尾再统一提交
            wandb.log({
                "train/tb_loss": tb_loss.item(),
                "train/trp_loss": trp_loss.item(),
                "train/total_loss": total_loss.item(),
            }, commit=False)

        
        if return_outputs:
            # 构造一个简单的 Namespace/dict 结构，保存 tb 和 trp 的 logits
            outputs = {
                "tb_logits_per_image": tb_logits_per_image,
                "tb_logits_per_text":  tb_logits_per_text,
                "trp_logits_per_image": trp_logits_per_image,
                "trp_logits_per_text":  trp_logits_per_text,
            }
            return total_loss, outputs
        return total_loss

    # NEW: override prediction_step to avoid unexpected kwargs during evaluation
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Custom evaluation step that reuses compute_loss logic to prevent forward() kwargs errors."""
        model.eval()
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # For simplicity, we do not return logits/labels for now.
        return (loss.detach(), None, None)


class CLIPRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, processor):
        self.dataset = dataset_dict
        self.processor = processor
        # 每个样本有3个tb + 3个trp = 6个caption，交叉组合生成3*3=9个图像-文本对
        self.captions_per_image = 9  # 3个tb * 3个trp = 9个组合
    
    def __len__(self):
        # 每个原始样本生成9个caption pair
        return len(self.dataset) * self.captions_per_image
    
    def __getitem__(self, idx):
        # 计算原始样本索引和caption组合索引
        original_idx = idx // self.captions_per_image
        pair_idx = idx % self.captions_per_image
        
        item = self.dataset[original_idx]
        
        # 读取图像
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")

        # 随机生成图像，用于测试
        # random_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # image = Image.fromarray(random_image)
        
        # 获取tb和trp列表
        tb_captions = item["tb"]  # 3个基础caption
        trp_captions = item["trp"]  # 3个推理caption
        
        # 计算tb和trp的索引（3x3组合）
        tb_idx = pair_idx // 3  # 0, 0, 0, 1, 1, 1, 2, 2, 2
        trp_idx = pair_idx % 3   # 0, 1, 2, 0, 1, 2, 0, 1, 2
        
        # 获取对应的tb和trp caption
        tb_caption = tb_captions[tb_idx]
        trp_caption = trp_captions[trp_idx]
        
        # 使用CLIP处理器分别处理图像和两种文本
        # 处理tb caption
        tb_encoding = self.processor(
            text=[tb_caption], 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        
        # 处理trp caption
        trp_encoding = self.processor(
            text=[trp_caption], 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        
        # 构建返回的batch，只包含必要的数据
        batch = {
            # 图像信息
            "pixel_values": tb_encoding["pixel_values"].squeeze(0),
            
            # TB文本信息
            "tb_input_ids": tb_encoding["input_ids"].squeeze(0),
            "tb_attention_mask": tb_encoding["attention_mask"].squeeze(0),
            
            # TRP文本信息
            "trp_input_ids": trp_encoding["input_ids"].squeeze(0),
            "trp_attention_mask": trp_encoding["attention_mask"].squeeze(0)
        }
        
        return batch


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = accelerator.is_main_process

    
    # 打印分布式训练信息
    main_print("="*60)
    main_print("🚀 CLIP-R Training Configuration")
    main_print("="*60)
    main_print(f"🔧 Training setup:")
    main_print(f"   - Distributed training: {accelerator.num_processes > 1}")
    main_print(f"   - Number of processes: {accelerator.num_processes}")
    main_print(f"   - Mixed precision: {accelerator.mixed_precision}")
    if accelerator.num_processes > 1:
        main_print(f"   - Current process rank: {accelerator.process_index}")
    
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
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # ================================ 数据集配置 ================================
    # 读取parquet数据集
    main_print(f"\n📊 Dataset Configuration:")
    main_print(f"   - Loading from: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)
    main_print(f"   - Total samples: {len(df)}")
    
    # 验证数据格式
    required_columns = ["image_path", "tb", "trp"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 将DataFrame转换为字典格式供CLIPRDataset使用
    dataset_dict = df.to_dict('records')
    
    # 分割训练、验证和测试集 (8:1:1)
    if args.use_split:
        # 首先分出训练集和临时集 (8:2)
        train_data, temp_data = train_test_split(
            dataset_dict, 
            test_size=0.2,  # 20% 用于验证+测试
            random_state=42
        )
        
        # 然后将临时集分为验证集和测试集 (1:1)
        eval_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,  # 50% 的临时数据作为测试集，50% 作为验证集
            random_state=42
        )
        
        main_print(f"   - Dataset split (8:1:1): {len(train_data)} train, {len(eval_data)} eval, {len(test_data)} test")
    else:
        train_data = dataset_dict
        eval_data = None
        test_data = None
        main_print(f"   - Using full dataset for training: {len(train_data)} samples")

    # 创建数据集
    train_dataset = CLIPRDataset(train_data, processor)
    eval_dataset = CLIPRDataset(eval_data, processor) if eval_data else None
    
    main_print(f"   - Train dataset size: {len(train_dataset)} (with 9x augmentation)")
    if eval_dataset:
        main_print(f"   - Eval dataset size: {len(eval_dataset)} (with 9x augmentation)")
    
    # 验证数据样本
    main_print(f"\n🔍 Data Validation:")
    sample = train_dataset[0]
    main_print(f"   - Sample keys: {list(sample.keys())}")
    main_print(f"   - TB Input IDs shape: {sample['tb_input_ids'].shape}")
    main_print(f"   - TRP Input IDs shape: {sample['trp_input_ids'].shape}")
    main_print(f"   - Pixel values shape: {sample['pixel_values'].shape}")
    main_print(f"   - ✅ Triplet data format validated: (image, tb_text, trp_text)")
    
    # ================================ 训练参数配置 ================================
    # 计算总步数来确定实际的logging、save、eval步数
    total_samples = len(train_dataset)
    steps_per_epoch = total_samples // (args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes)
    total_steps = steps_per_epoch * args.epochs
    
    main_print(f"\n⚡ Training Schedule:")
    main_print(f"   - Total samples: {total_samples}")
    main_print(f"   - Per-device batch size: {args.batch_size}")
    main_print(f"   - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    main_print(f"   - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    main_print(f"   - Steps per epoch: {steps_per_epoch}")
    main_print(f"   - Total training steps: {total_steps}")
    
    # 根据策略计算实际步数
    # Logging steps
    if args.logging_strategy == "epoch":
        logging_steps = steps_per_epoch
        logging_strategy = "epoch"
    elif args.logging_strategy == "ratio":
        logging_steps = max(1, int(total_steps * args.logging_ratio))
        logging_strategy = "steps"
    else:  # steps
        logging_steps = args.logging_steps
        logging_strategy = "steps"
    
    # Save steps
    if args.save_strategy == "epoch":
        save_steps = steps_per_epoch
        save_strategy = "epoch"
    elif args.save_strategy == "ratio":
        save_steps = max(1, int(total_steps * args.save_ratio))
        save_strategy = "steps"
    else:  # steps
        save_steps = args.save_steps
        save_strategy = "steps"
    
    # Eval steps
    if args.eval_strategy == "epoch":
        eval_steps = steps_per_epoch
        eval_strategy = "epoch"
    elif args.eval_strategy == "ratio":
        eval_steps = max(1, int(total_steps * args.eval_ratio))
        eval_strategy = "steps"
    else:  # steps
        eval_steps = args.eval_steps
        eval_strategy = "steps"
    
    main_print(f"\n📝 Logging & Evaluation Schedule:")
    main_print(f"   - Logging: every {logging_steps} steps ({args.logging_strategy})")
    main_print(f"   - Saving: every {save_steps} steps ({args.save_strategy})")
    main_print(f"   - Evaluation: every {eval_steps} steps ({args.eval_strategy})")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,  # 保留更多检查点
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
        remove_unused_columns=False,
        # 分布式训练配置
        ddp_find_unused_parameters=False,  # 关闭unused parameters检测，提高性能
        dataloader_drop_last=True,            # 丢弃训练集最后一个不满 batch
    )
    
    main_print(f"\n🎯 Loss Configuration:")
    main_print(f"   - TB loss weight: {args.tb_alpha:.3f}")
    main_print(f"   - TRP loss weight: {1.0 - args.tb_alpha:.3f}")
    
    main_print(f"\n🚀 Starting Training...")
    main_print("="*60)
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        tb_alpha=args.tb_alpha,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[BestModelCallback()]
    )

    trainer.train()

    # 手动保存 best model
    best_model_path = args.best_model_dir
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    main_print(f"\n💾 Best model saved to: {best_model_path}")
 

    return best_model_path


def push_to_hub(best_model_path, repo_name):
    # 只有主进程执行推送操作
    if accelerator.is_main_process:
        main_print(f"\n🤗 Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        # 自动上传 best model 到 hub
        model = CLIPModel.from_pretrained(best_model_path)
        processor = CLIPProcessor.from_pretrained(best_model_path)

        model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)

        main_print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
    # No need to print skip message for non-main processes

    # 等待所有进程同步
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    main_print(f"Arguments: {args}")
    
    best_model_path = train_clip(args)
    
    # 推送到HuggingFace Hub
    if args.push_to_hub:
        push_to_hub(best_model_path, args.hub_model_name)
    
    # Distributed training cleanup
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        main_print("\n🎉 All processes have completed successfully!")


if __name__ == "__main__":
    main()
