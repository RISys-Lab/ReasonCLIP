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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    # Training parameters
    
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", 
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./weights/clip_R_finetune", 
                        help="Output directory")
    parser.add_argument("--best_model_dir", type=str, default="./weights/clip_R_best_model", 
                        help="Directory to save the best model")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, 
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=25, 
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Steps to save checkpoints")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Evaluation steps")
    parser.add_argument("--run_name", type=str, default="clip_R_finetune", 
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
    parser.add_argument("--hub_model_name", type=str, default="clip-R-L14", 
                        help="Model name on the Hub")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="TBD",
                      help="Dataset name on HuggingFace Hub")

    # wandb parameters
    parser.add_argument("--wandb_project", type=str, default="clip-R",
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

            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            
            # 手动记录到 Wandb
            if is_main_process and (args.report_to == "wandb" or (isinstance(args.report_to, list) and "wandb" in args.report_to)):
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval_loss": eval_loss}, step=state.global_step)
                    print(f"已手动记录 eval_loss={eval_loss:.4f} 到 Wandb (step={state.global_step})")
            
            # 检查是否为新的最佳模型
            if eval_loss < self.best_eval_loss:
                print(f"\n>>> eval_loss: {eval_loss:.4f}\n")
                self.best_eval_loss = eval_loss
                print(f"\n*** New best model: {state.global_step}, Loss: {self.best_eval_loss:.4f} ***\n")

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 调用模型，不传 labels
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"]
        )
        logits_per_image = outputs.logits_per_image
        logits_per_text  = outputs.logits_per_text

        # 构造正确的 labels：0..batch_size-1
        bs = logits_per_image.size(0)
        labels = torch.arange(bs, device=logits_per_image.device)

        # 计算两边对比损失
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss   = (loss_i + loss_t) / 2

        return (loss, outputs) if return_outputs else loss
    
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
            
        text = item["caption"]  # 确认字段名是否正确
        
        # 使用CLIP处理器处理图像和文本
        encoding = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt",
            padding="max_length",
            truncation=True
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
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Download dataset if specified
    if hasattr(args, 'dataset_name') and args.dataset_name:
        try:
            # 尝试加载数据集
            raw_ds = load_dataset(args.dataset_name)
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
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise ValueError(f"Failed to load dataset {args.dataset_name}: {e}")
    else:
        raise ValueError("Please specify a dataset name using --dataset_name argument.")



    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
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
        max_grad_norm=args.max_grad_norm
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",  # 使用验证损失作为指标
        # greater_is_better=False,       # 损失越小越好
    )
    

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[BestModelCallback()]
    )

    trainer.train()

    # 手动保存 best model
    if not dist.is_initialized() or dist.get_rank() == 0:
        best_model_path = args.best_model_dir
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        print(f"Best model saved to {best_model_path}")
    else:
        best_model_path = args.best_model_dir
        print(f"Skipping saving best model for rank {dist.get_rank()}")

    return best_model_path


def push_to_hub(best_model_path, repo_name):
    # 检查当前进程的rank
    import os
    import torch.distributed as dist
    
    # 只有rank0进程执行推送操作
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        # 自动上传 best model 到 hub
        model = CLIPModel.from_pretrained(best_model_path)
        processor = CLIPProcessor.from_pretrained(best_model_path)

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
    best_model_path = train_clip(args)
    
    if args.push_to_hub and args.hub_username:
        repo_name = f"{args.hub_username}/{args.hub_model_name}"
        push_to_hub(best_model_path, repo_name)


    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print("All processes have completed successfully.")
