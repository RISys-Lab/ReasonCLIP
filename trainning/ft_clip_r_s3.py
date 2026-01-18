import os
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    SiglipProcessor,
    SiglipModel,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
import torch
import torch.nn as nn
from accelerate import Accelerator
import numpy as np 
from typing import Optional, List
import math
import glob
from torch.optim import AdamW
import torch.distributed as dist

# =========================== NEW IMPORT ===========================
from peft import LoraConfig, get_peft_model, TaskType
# ==================================================================

# 初始化 accelerator
accelerator = Accelerator()

def main_print(*args, **kwargs):
    """只在主进程打印的函数"""
    if accelerator.is_main_process:
        print(*args, **kwargs)

# 非主进程立即禁用 Wandb
if not accelerator.is_main_process:
    os.environ["WANDB_DISABLED"] = "true"

import wandb
import argparse
from transformers import TrainerCallback
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for CLIP/SigLIP")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "siglip"],
                        help="Model type: 'clip' or 'siglip'")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", 
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./weights/unifire_clip_lora", 
                        help="Output directory")
    parser.add_argument("--best_model_dir", type=str, default="./weights/unifire_clip_lora_best", 
                        help="Directory to save the best model")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # 针对 CLIP/SigLIP 的常用 target modules
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                        default=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                        help="Target modules for LoRA")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--default_lr", type=float, default=1e-4, help="Learning rate for LoRA adapters")
    
    # 兼容旧脚本的参数（LoRA模式下可能不生效，但保留以防报错）
    parser.add_argument("--visual_lr", type=float, default=None) 
    parser.add_argument("--text_lr", type=float, default=None) 
    parser.add_argument("--logit_scale_lr", type=float, default=None) 

    parser.add_argument("--fp16", action="store_true", help="Mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--flash_attn", action="store_true")
    
    # Logging/Save/Eval
    parser.add_argument("--logging_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"])
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--logging_ratio", type=float, default=0.02)
    
    parser.add_argument("--save_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_ratio", type=float, default=0.1)
    
    parser.add_argument("--eval_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"])
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--run_name", type=str, default="clip-lora-unifire")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_username", type=str, default="fesvhtr")
    parser.add_argument("--hub_model_name", type=str, default="clip-lora-test")
    
    # Dataset
    parser.add_argument("--parquet_files", type=str, nargs="+", required=True,
                        help="List of parquet files, each will train a separate LoRA adapter")
    parser.add_argument("--use_split", action="store_true")
    parser.add_argument("--holdout_ratio", type=float, default=0.002)
    
    default_workers = min(8, os.cpu_count() // 2)
    parser.add_argument("--num_workers", type=int, default=default_workers)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="clip-unifire")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log", action="store_true")
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def compute_strategy_steps(strategy, ratio_value, steps_value, ratio_multiplier, steps_per_epoch):
    if strategy == "epoch":
        return steps_per_epoch, "epoch"
    elif strategy == "ratio":
        return max(1, int(ratio_multiplier * ratio_value)), "steps"
    else:  # steps
        return steps_value, "steps"

class BestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float('inf')
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            is_main_process = accelerator.is_main_process
            
            if is_main_process and (args.report_to == "wandb" or (isinstance(args.report_to, list) and "wandb" in args.report_to)):
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval_loss": eval_loss}, step=state.global_step)
            
            if eval_loss < self.best_eval_loss:
                main_print(f"\n>>> eval_loss: {eval_loss:.4f}\n")
                self.best_eval_loss = eval_loss
                main_print(f"\n*** New best model: {state.global_step}, Loss: {self.best_eval_loss:.4f} ***\n")


class CLIPTrainer(Trainer):
    def __init__(self, model_type="clip", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.backbone = None
        # 注意：这里移除了 orig_model，因为 LoRA 本身就是正则化

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.backbone is None:
            self.backbone = model.module if hasattr(model, "module") else model
        
        def gather_with_local_grad(x, accelerator):
            if accelerator.num_processes == 1:
                return x
            B = x.size(0)
            gathered = accelerator.gather(x.detach())
            parts = []
            for i in range(accelerator.num_processes):
                if i == accelerator.process_index:
                    parts.append(x)
                else:
                    s, e = i * B, (i + 1) * B
                    parts.append(gathered[s:e])
            return torch.cat(parts, dim=0)

        device = inputs["pixel_values"].device

        # LoRA 模式下，model(inputs) 调用的是 PEFT 包装后的模型
        # 我们依然可以使用 backbone 的特定方法，或者直接调用 forward
        # 但这里为了保持逻辑一致，我们手动调用 get_features
        # PEFT model 会自动路由到 base model 的方法
        
        image_features = self.backbone.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.backbone.get_text_features(
            input_ids=inputs["text_input_ids"],
            attention_mask=inputs["text_attention_mask"],
        )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        with torch.no_grad():
            if hasattr(self.backbone, "logit_scale"):
                self.backbone.logit_scale.data.clamp_(max=math.log(100.0))
        
        # 兼容 PEFT 包装带来的属性访问层级变化
        # 有时 PEFT 会把原始模型放在 model.base_model.model 中
        # 但 getattr 会递归查找，或者直接用 self.backbone 访问即可
        logit_scale = self.backbone.logit_scale.exp() if hasattr(self.backbone, "logit_scale") else 1.0

        logit_bias = getattr(self.backbone, "logit_bias", None)
        if logit_bias is not None:
            bias = logit_bias.to(image_features.dtype)
        else:
            bias = 0.0

        B = image_features.size(0)
        rank = accelerator.process_index
        all_image = gather_with_local_grad(image_features, accelerator)
        all_trp   = gather_with_local_grad(text_features, accelerator)

        labels_global = torch.arange(B, device=device) + rank * B

        trp_logits_per_image = logit_scale * (image_features     @ all_trp.t()) + bias
        trp_logits_per_text  = logit_scale * (text_features  @ all_image.t()) + bias
        
        trp_loss = 0.5 * (
            F.cross_entropy(trp_logits_per_image, labels_global) +
            F.cross_entropy(trp_logits_per_text,  labels_global)
        )

        del trp_logits_per_image, trp_logits_per_text

        contrastive_loss = trp_loss
        total_loss = contrastive_loss

        # 注意：移除了 L2 Regularization，因为 LoRA 不需要

        if accelerator.is_main_process and "wandb" in self.args.report_to:
            import wandb
            wandb.log(
                {
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/total_loss": total_loss.item(),
                },
                commit=False,
            )

        if return_outputs:
            with torch.no_grad():
                trp_logits_view = (image_features @ all_trp.t()).detach()
            outputs = {
                "trp_image_text_logits": trp_logits_view,
            }
            return total_loss, outputs

        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), None, None)

class CLIPRProDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, processor):
        self.dataset = dataset_dict
        self.processor = processor
        proc_name = processor.__class__.__name__.lower()
        if "siglip" in proc_name:
            self.text_max_len = 64
        else:
            self.text_max_len = 77
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        trp_caption = item["trp"]
        img_enc = self.processor(images=image, return_tensors="pt")
        trp_enc = self.processor(text=trp_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.text_max_len)
        return {
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "text_input_ids": trp_enc["input_ids"].squeeze(0),
            "text_attention_mask": trp_enc.get("attention_mask", torch.ones_like(trp_enc["input_ids"])).squeeze(0),
        }


def get_adapter_id_from_parquet(parquet_path: str) -> str:
    """从 parquet 文件名提取最后一个字母作为 adapter ID"""
    # e.g. "/path/to/data_A.parquet" -> "A"
    basename = os.path.basename(parquet_path)  # "data_A.parquet"
    name_without_ext = os.path.splitext(basename)[0]  # "data_A"
    adapter_id = name_without_ext[-1]  # "A"
    return adapter_id


def train_clip(args):
    is_main_process = accelerator.is_main_process
    model_name = args.model_name
    model_type = args.model_type

    main_print("="*60)
    main_print("🚀 CLIP-R Training Configuration (LoRA - Multi-Adapter)")
    main_print("="*60)
    main_print(f"🔧 Training setup:")
    main_print(f"   - Model type: {model_type.upper()}")
    main_print(f"   - Model name: {model_name}")
    main_print(f"   - LoRA Rank: {args.lora_r}")
    main_print(f"   - LoRA Alpha: {args.lora_alpha}")
    main_print(f"   - LoRA Target Modules: {args.lora_target_modules}")
    main_print(f"   - Number of adapters to train: {len(args.parquet_files)}")

    # 统一 run_id
    from datetime import datetime
    import torch.distributed as dist

    if accelerator.is_main_process:
        run_id = f"run_{datetime.now().strftime('%m%d_%H%M%S')}"
    else:
        run_id = None

    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        obj_list = [run_id]
        dist.broadcast_object_list(obj_list, src=0)
        run_id = obj_list[0]

    save_root = os.path.join(args.output_dir, run_id)
    original_run_name = args.run_name

    # 创建总目录
    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
    accelerator.wait_for_everyone()

    # 加载 processor（只需要加载一次）
    if model_type == "clip":
        processor = CLIPProcessor.from_pretrained(model_name)
    elif model_type == "siglip":
        processor = SiglipProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    adapter_paths = []

    # ==================== FOR LOOP: 训练每个 adapter ====================
    for parquet_idx, parquet_file in enumerate(args.parquet_files):
        adapter_id = get_adapter_id_from_parquet(parquet_file)
        
        main_print("\n" + "="*60)
        main_print(f"📁 Training Adapter [{parquet_idx+1}/{len(args.parquet_files)}]: {adapter_id}")
        main_print(f"   Parquet file: {parquet_file}")
        main_print("="*60)

        # 为每个 adapter 创建独立的子文件夹
        adapter_save_dir = os.path.join(save_root, adapter_id)
        adapter_ckpt_dir = os.path.join(adapter_save_dir, "checkpoints")
        
        if accelerator.is_main_process:
            os.makedirs(adapter_save_dir, exist_ok=True)
            os.makedirs(adapter_ckpt_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # 每次循环重新加载 Base Model（确保从头训练）
        if model_type == "clip":
            model = CLIPModel.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            )
        elif model_type == "siglip":
            model = SiglipModel.from_pretrained(
            model = SiglipModel.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
            )

        # =========================== LoRA SETUP ===========================
        modules_to_save = ["logit_scale"]
        if model_type == "siglip":
            modules_to_save.append("logit_bias")

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=modules_to_save
        )

        model = get_peft_model(model, peft_config)
        
        if is_main_process and parquet_idx == 0:  # 只在第一个 adapter 打印参数信息
            model.print_trainable_parameters()
        # ==================================================================

        # Wandb 初始化（每个 adapter 一个 run）
        adapter_run_name = f"{original_run_name}_{run_id.replace('run_', '')}_{adapter_id}"
        
        if args.wandb_log and is_main_process:
            if parquet_idx == 0:
                wandb.login()
            # 如果已经有 run，先结束
            if wandb.run is not None:
                wandb.finish()
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=adapter_run_name)
        else:
            os.environ["WANDB_DISABLED"] = "true"

        # 数据集加载
        from datasets import load_dataset

        main_print(f"\n📊 Dataset Configuration for Adapter {adapter_id}:")
        hf_ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train", keep_in_memory=False)
        
        if args.use_split:
            split1 = hf_ds.train_test_split(test_size=0.2, seed=42)
            train_hf = split1["train"]
            tmp = split1["test"].train_test_split(test_size=0.5, seed=42)
            eval_hf, test_hf = tmp["train"], tmp["test"]
        else:
            if args.holdout_ratio > 0:
                split = hf_ds.train_test_split(test_size=args.holdout_ratio, seed=42)
                train_hf, eval_hf = split["train"], split["test"]
            else:
                train_hf, eval_hf = hf_ds, None

        train_dataset = CLIPRProDataset(train_hf, processor)
        eval_dataset = CLIPRProDataset(eval_hf, processor) if eval_hf else None
        
        main_print(f"   - Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            main_print(f"   - Eval dataset size: {len(eval_dataset)}")

        # 训练参数计算
        total_samples = len(train_dataset)
        steps_per_epoch = total_samples // (args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes)
        total_steps = steps_per_epoch * args.epochs
        
        logging_steps, logging_strategy = compute_strategy_steps(args.logging_strategy, args.logging_ratio, args.logging_steps, total_steps, steps_per_epoch)
        save_steps, save_strategy = compute_strategy_steps(args.save_strategy, args.save_ratio, args.save_steps, total_steps, steps_per_epoch)
        eval_steps, eval_strategy = compute_strategy_steps(args.eval_strategy, args.eval_ratio, args.eval_steps, total_steps, steps_per_epoch)

        training_args = TrainingArguments(
            output_dir=adapter_ckpt_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bf16=args.bf16,
            fp16=args.fp16 and (not args.bf16),
            num_train_epochs=args.epochs,
            learning_rate=args.default_lr,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_total_limit=args.save_total_limit,
            report_to="wandb" if args.wandb_log and is_main_process else "none",
            run_name=adapter_run_name,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=True,
            # load_best_model_at_end=True,  # 注释掉：不再保存最佳模型
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
            dataloader_num_workers=args.num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_drop_last=True,
            deepspeed=args.deepspeed,
            seed=42,
            data_seed=42,
        )

        # 优化器
        main_print(f"🔧 Setting up optimizer for LoRA (Adapter {adapter_id})...")
        main_print(f"   - LoRA Learning Rate: {args.default_lr}")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=args.default_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        trainer = CLIPTrainer(
            model=model,
            args=training_args,
            model_type=model_type,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # callbacks=[BestModelCallback()],  # 注释掉：不再使用 best model callback
            optimizers=(optimizer, None),
        )

        # 开始训练
        main_print(f"🚀 Starting training for Adapter {adapter_id}...")
        trainer.train()
        
        accelerator.wait_for_everyone()

        # 保存最终的 LoRA adapter（训练结束后的状态）
        final_adapter_path = os.path.join(adapter_save_dir, "final_adapter")
        
        if accelerator.is_main_process:
            trainer.save_model(final_adapter_path)
            processor.save_pretrained(final_adapter_path)
            main_print(f"\n💾 Adapter {adapter_id} saved to: {final_adapter_path}")

        accelerator.wait_for_everyone()
        adapter_paths.append(final_adapter_path)

        # 清理内存
        del model, trainer, optimizer, train_dataset, eval_dataset
        torch.cuda.empty_cache()

    # 结束 wandb
    if args.wandb_log and is_main_process and wandb.run is not None:
        wandb.finish()

    main_print("\n" + "="*60)
    main_print("✅ All adapters trained successfully!")
    main_print("="*60)
    for path in adapter_paths:
        main_print(f"   - {path}")

    return adapter_paths


def push_to_hub(adapter_paths, base_repo_name, model_type="clip"):
    """Push all adapters to HuggingFace Hub, each with its own repo suffix"""
    if accelerator.is_main_process:
        from peft import PeftModel, PeftConfig
        
        for adapter_path in adapter_paths:
            # 从路径提取 adapter ID (最后一级目录的父目录名)
            # e.g. "/path/run_xxx/A/final_adapter" -> "A"
            adapter_id = os.path.basename(os.path.dirname(adapter_path))
            repo_name = f"{base_repo_name}-{adapter_id}"
            
            main_print(f"\n🤗 Pushing adapter {adapter_id} to HuggingFace Hub: {repo_name}")
            
            if model_type == "clip":
                config = PeftConfig.from_pretrained(adapter_path)
                base_model = CLIPModel.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                processor = CLIPProcessor.from_pretrained(adapter_path)
            elif model_type == "siglip":
                config = PeftConfig.from_pretrained(adapter_path)
                base_model = SiglipModel.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                processor = SiglipProcessor.from_pretrained(adapter_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Push adapter only
            model.push_to_hub(repo_name)
            processor.push_to_hub(repo_name)

            main_print(f"✅ LoRA Adapter {adapter_id} pushed to: https://huggingface.co/{repo_name}")
            
            # 清理内存
            del model, base_model
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    main_print(f"Arguments: {args}")
    
    adapter_paths = train_clip(args)
    
    if args.push_to_hub:
        push_to_hub(adapter_paths, args.hub_model_name, args.model_type)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        main_print("\n🎉 All processes have completed successfully!")


if __name__ == "__main__":
    main()