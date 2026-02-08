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
    TrainerCallback,
)
import torch.nn.functional as F
import torch.nn as nn
from accelerate import Accelerator
import numpy as np 
from typing import Optional, List
import math
import glob
from torch.optim import AdamW
import torch.distributed as dist
import argparse
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb

# 初始化 accelerator
accelerator = Accelerator()

def main_print(*args, **kwargs):
    """只在主进程打印的函数"""
    if accelerator.is_main_process:
        print(*args, **kwargs)

# 非主进程立即禁用 Wandb
if not accelerator.is_main_process:
    os.environ["WANDB_DISABLED"] = "true"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "siglip"],
                        help="Model type: 'clip' or 'siglip'")
    parser.add_argument("--use_sigmoid_loss", action="store_true",
                        help="Use SigLIP logistic loss when model_type=siglip")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14", 
                        help="Pre-trained model name")
    parser.add_argument("--processor_name", type=str, default=None,
                        help="Processor name/path (default: same as model_name)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./weights/unifire_clip_finetune", 
                        help="Output directory")
    parser.add_argument("--best_model_dir", type=str, default="./weights/unifire_clip_best_model", 
                        help="Directory to save the best model")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    
    # Learning Rates
    parser.add_argument("--default_lr", type=float, default=1e-4, 
                            help="Learning rate")
    parser.add_argument("--visual_lr", type=float, default=None, 
                            help="Visual encoder learning rate")
    parser.add_argument("--text_lr", type=float, default=None, 
                            help="Text encoder learning rate")
    parser.add_argument("--logit_scale_lr", type=float, default=None, 
                            help="Logit scale learning rate")

    # Precision & Optimization
    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (Ampere+ GPUs)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed JSON config (ZeRO, etc.)")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use FlashAttention-2 backend")
    
    # Logging parameters
    parser.add_argument("--logging_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Logging strategy")
    parser.add_argument("--logging_steps", type=int, default=25, 
                        help="Logging steps")
    parser.add_argument("--logging_ratio", type=float, default=0.02,
                        help="Logging ratio")
    
    # Save parameters
    parser.add_argument("--save_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="Steps to save checkpoints")
    parser.add_argument("--save_ratio", type=float, default=0.1,
                        help="Save ratio")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Total number of checkpoints to save")
    
    # Evaluation parameters
    parser.add_argument("--eval_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Evaluation steps")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Evaluation ratio")
    
    # Misc
    parser.add_argument("--run_name", type=str, default="clip-finetune-unifire", 
                        help="Experiment name")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--use_l2_reg", action="store_true",
                        help="Enable L2 regularization to original pretrained weights")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Max gradient norm for gradient clipping")
    
    # Hub push parameters
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="fesvhtr", 
                        help="HuggingFace username")
    parser.add_argument("--hub_model_name", type=str, default="clip-finetune", 
                        help="Model name on the Hub")
    
    # Dataset parameters
    parser.add_argument("--parquet_files", type=str, nargs="+", required=True,
                        help="Paths to one or more parquet files")
    parser.add_argument("--use_split", action="store_true",
                        help="Whether to split dataset into train:eval:test = 8:1:1")
    parser.add_argument("--holdout_ratio", type=float, default=0.002,
                    help="Fraction reserved for eval only (e.g., 0.002 = 0.2%)")
    
    default_workers = min(8, os.cpu_count() // 2)
    parser.add_argument("--num_workers", type=int, default=default_workers,
                        help="Number of workers for data loading")

    # wandb parameters
    parser.add_argument("--wandb_project", type=str, default="clip-unifire",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity name")
    parser.add_argument("--wandb_log", action="store_true",
                        help="Enable wandb logging")
    
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")

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
    def __init__(
        self,
        model_type="clip",
        orig_model=None,
        use_sigmoid_loss=False,
        use_l2_reg=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_type = model_type  # "clip" 或 "siglip"
        self.use_sigmoid_loss = use_sigmoid_loss
        self.use_l2_reg = use_l2_reg
        self.orig_model = orig_model
        self.orig_state = None
        self.backbone = None  # 延迟初始化
    
    def _initialize_l2_reg(self, model=None):
        """在模型移动到设备后，安全地初始化原始权重状态"""
        if self.orig_state is None and self.orig_model is not None:
            if model is not None:
                backbone = model.module if hasattr(model, "module") else model
            elif self.backbone is not None:
                backbone = self.backbone
            else:
                backbone = self.model.module if hasattr(self.model, "module") else self.model
            
            device = next(backbone.parameters()).device
            orig_device = next(self.orig_model.parameters()).device
            
            if orig_device != device:
                self.orig_model = self.orig_model.to(device)
            
            self.orig_state = {
                n: p.detach().clone().to(device, dtype=torch.float32)
                for n, p in self.orig_model.named_parameters()
            }
            main_print(f"[L2 Reg] Initialized orig_state on device: {device}")

    @staticmethod
    def _siglip_logistic_loss(
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        SigLIP 风格的 logistic loss
        """
        B = logits.size(0)
        device = logits.device

        # 构造 +1 / -1 的 label matrix
        label_matrix = logits.new_full(logits.shape, -1.0)
        row_idx = torch.arange(B, device=device)
        label_matrix[row_idx, labels] = 1.0

        # 对所有 pair 做 -log σ(z_ij * logit_ij) 的平均
        loss = -F.logsigmoid(label_matrix * logits).sum() / logits.size(0)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 动态获取 backbone
        if self.backbone is None:
            self.backbone = model.module if hasattr(model, "module") else model
        
        if self.use_l2_reg and self.orig_model is not None and self.orig_state is None:
            self._initialize_l2_reg(model)

        # ---- helper: gather across GPUs but keep local slice with gradient ----
        def gather_with_local_grad(x, accelerator):
            if accelerator.num_processes == 1:
                return x
            B = x.size(0)
            gathered = accelerator.gather(x.detach()) # [world*B, D], no grad
            parts = []
            for i in range(accelerator.num_processes):
                if i == accelerator.process_index:
                    parts.append(x) # keep grad for local slice
                else:
                    s, e = i * B, (i + 1) * B
                    parts.append(gathered[s:e])
            return torch.cat(parts, dim=0)

        # ---- unpack ----
        device = inputs["pixel_values"].device

        # ---- forward encoders ----
        image_features = self.backbone.get_image_features(pixel_values=inputs["pixel_values"])
        trp_text_features = self.backbone.get_text_features(
            input_ids=inputs["trp_input_ids"],
            attention_mask=inputs["trp_attention_mask"],
        )
        image_features = F.normalize(image_features, dim=-1)
        trp_text_features = F.normalize(trp_text_features, dim=-1)

        # ---- temperature (clamp to avoid blow-up in large-batch) ----
        with torch.no_grad():
            if hasattr(self.backbone, "logit_scale"):
                self.backbone.logit_scale.data.clamp_(max=math.log(50.0))
        logit_scale = self.backbone.logit_scale.exp() if hasattr(self.backbone, "logit_scale") else 1.0

        if self.use_sigmoid_loss and self.model_type == "siglip":
            logit_bias = getattr(self.backbone, "logit_bias", None)
            bias = logit_bias.to(image_features.dtype) if logit_bias is not None else 0.0
        else:
            bias = 0.0 

        # ---- cross-GPU gather ----
        B = image_features.size(0)
        rank = accelerator.process_index
        all_image = gather_with_local_grad(image_features, accelerator) 
        all_trp   = gather_with_local_grad(trp_text_features, accelerator)

        # global labels: shift by the local slice offset
        labels_global = torch.arange(B, device=device) + rank * B 

        # ---- Contrastive Branch ----
        trp_logits_per_image = logit_scale * (image_features     @ all_trp.t()) + bias
        trp_logits_per_text  = logit_scale * (trp_text_features  @ all_image.t()) + bias

        if self.use_sigmoid_loss and self.model_type == "siglip":
            trp_loss = self._siglip_logistic_loss(trp_logits_per_image, labels_global)
        else:
            trp_loss = 0.5 * (
                F.cross_entropy(trp_logits_per_image, labels_global) +
                F.cross_entropy(trp_logits_per_text,  labels_global)
            )

        del trp_logits_per_image, trp_logits_per_text

        # ---- combine ----
        total_loss = trp_loss

        # ---- L2 Regularization ----
        if self.use_l2_reg and self.orig_state is not None:
            beta = 1e-5 
            l2_reg = torch.zeros((), device=device, dtype=torch.float32)
            for name, p in self.backbone.named_parameters():
                if ("vision_model." in name) or ("text_model." in name):
                    if ("projection" in name) or ("logit_scale" in name):
                        continue
                    if name in self.orig_state:
                        p0 = self.orig_state[name]
                        l2_reg = l2_reg + (p.float() - p0.to(p.device)).pow(2).sum()
            total_loss = total_loss + beta * l2_reg

        # ---- optional logging ----
        if accelerator.is_main_process and "wandb" in self.args.report_to:
            import wandb
            wandb.log(
                {
                    "train/contrastive_loss": trp_loss.item(),
                    "train/total_loss": total_loss.item(),
                },
                commit=False,
            )

        # ---- optional outputs ----
        if return_outputs:
            with torch.no_grad():
                trp_logits_view = (image_features @ all_trp.t()).detach()
            outputs = {
                "trp_image_text_logits": trp_logits_view,
            }
            return total_loss, outputs

        return total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        model.eval()
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, None)


class CLIPRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, processor, lowercase_text=False):
        self.dataset = dataset_dict
        self.processor = processor
        self.lowercase_text = lowercase_text
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
        if self.lowercase_text and isinstance(trp_caption, str):
            trp_caption = trp_caption.lower()

        img_enc = self.processor(images=image, return_tensors="pt")
        trp_enc = self.processor(text=trp_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.text_max_len)
        
        return {
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "trp_input_ids": trp_enc["input_ids"].squeeze(0),
            "trp_attention_mask": trp_enc.get("attention_mask", torch.ones_like(trp_enc["input_ids"])).squeeze(0),
        }


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = accelerator.is_main_process

    model_name = args.model_name
    model_type = args.model_type

    # 打印分布式训练信息
    main_print("="*60)
    main_print("🚀 CLIP/SigLIP Training Configuration")
    main_print("="*60)
    main_print(f"🔧 Training setup:")
    main_print(f"   - Model type: {model_type.upper()}")
    main_print(f"   - Model name: {model_name}")
    main_print(f"   - Number of processes: {accelerator.num_processes}")
    main_print(f"   - Mixed precision: {accelerator.mixed_precision}")

    # 统一 run_id
    if accelerator.is_main_process:
        run_id = f"run_{datetime.now().strftime('%m%d_%H%M%S')}"
    else:
        run_id = None

    if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
        obj_list = [run_id]
        dist.broadcast_object_list(obj_list, src=0)
        run_id = obj_list[0]

    save_root = os.path.join(args.output_dir, run_id)

    # 只主进程创建目录
    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_root, "finetune_weights"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "best_model"), exist_ok=True)
    accelerator.wait_for_everyone()

    args.run_name = f"{args.run_name}_{run_id.replace('run_', '')}"
    args.output_dir = os.path.join(save_root, "finetune_weights")
    args.best_model_dir = os.path.join(save_root, "best_model")

    # 初始化 WandB
    if args.wandb_log and is_main_process:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    # 加载模型
    if model_type == "clip":
        model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        orig_model = None
        if args.use_l2_reg:
            orig_model = CLIPModel.from_pretrained(model_name)
            for p in orig_model.parameters():
                p.requires_grad = False
        processor_name = args.processor_name or model_name
        processor = CLIPProcessor.from_pretrained(processor_name)
    elif model_type == "siglip":
        model = SiglipModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        orig_model = None
        if args.use_l2_reg:
            orig_model = SiglipModel.from_pretrained(model_name)
            for p in orig_model.parameters():
                p.requires_grad = False
        processor_name = args.processor_name or model_name
        processor = SiglipProcessor.from_pretrained(processor_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # ================================ 数据集配置 ================================
    main_print(f"\n📊 Dataset Configuration:")
    main_print(f"   - Loading from: {args.parquet_files}")

    files = sorted(args.parquet_files)
    hf_ds = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
        keep_in_memory=False
    )
    main_print(f"   - Total samples (rows): {len(hf_ds)}")

    if args.use_split:
        split1 = hf_ds.train_test_split(test_size=0.2, seed=42)
        train_hf = split1["train"]
        tmp     = split1["test"].train_test_split(test_size=0.5, seed=42)
        eval_hf, test_hf = tmp["train"], tmp["test"]
        main_print(f"   - Dataset split (8:1:1): {len(train_hf)} train, {len(eval_hf)} eval, {len(test_hf)} test")
    else:
        if args.holdout_ratio > 0:
            split = hf_ds.train_test_split(test_size=args.holdout_ratio, seed=42)
            train_hf, eval_hf = split["train"], split["test"]
            main_print(f"   - Holdout eval: {len(eval_hf)} ({args.holdout_ratio*100:.2f}%)")
        else:
            train_hf, eval_hf = hf_ds, None
            main_print(f"   - No eval holdout")

    lowercase_text = model_type == "siglip" and "siglip2" in str(model_name).lower()
    train_dataset = CLIPRDataset(train_hf, processor, lowercase_text=lowercase_text)
    eval_dataset  = CLIPRDataset(eval_hf, processor, lowercase_text=lowercase_text) if eval_hf else None

    main_print(f"   - Train dataset size: {len(train_dataset)}")
    
    # 验证数据样本
    main_print(f"\n🔍 Data Validation:")
    sample = train_dataset[0]
    main_print(f"   - Sample keys: {list(sample.keys())}")
    main_print(f"   - TRP Input IDs shape: {sample['trp_input_ids'].shape}")
    main_print(f"   - Pixel values shape: {sample['pixel_values'].shape}")
    main_print(f"   - ✅ Stage 2 data format validated: (image, trp_text)")
    
    # ================================ 训练参数配置 ================================
    total_samples = len(train_dataset)
    steps_per_epoch = total_samples // (args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes)
    total_steps = steps_per_epoch * args.epochs
    
    logging_steps, logging_strategy = compute_strategy_steps(
        args.logging_strategy, args.logging_ratio, args.logging_steps, total_steps, steps_per_epoch)
    save_steps, save_strategy = compute_strategy_steps(
        args.save_strategy, args.save_ratio, args.save_steps, total_steps, steps_per_epoch)
    eval_steps, eval_strategy = compute_strategy_steps(
        args.eval_strategy, args.eval_ratio, args.eval_steps, total_steps, steps_per_epoch)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb" if args.wandb_log and is_main_process else "none",
        run_name=args.run_name,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        deepspeed=args.deepspeed,
        seed=42,
        data_seed=42,
    )
    
    main_print(f"\n🎯 Loss Configuration:")
    main_print(f"   - Contrastive loss (TRP): 1.0")
    
    # ================================ 断点恢复 ================================
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            resume_from_checkpoint = args.resume_from_checkpoint
            main_print(f"\n🔄 Resuming from specified checkpoint: {resume_from_checkpoint}")
        else:
            main_print(f"❌ Specified checkpoint not found: {args.resume_from_checkpoint}")
            main_print("   Starting training from scratch...")

    main_print(f"\n🚀 Starting Training...")
    main_print("="*60)
    
    main_print(f"🔧 Setting up optimizer with different learning rates...")
    default_lr = args.default_lr
    visual_lr = args.visual_lr
    text_lr = args.text_lr
    logit_scale_lr = args.logit_scale_lr
    
    main_print(f"   - Default learning rate: {default_lr}")
    main_print(f"   - Visual learning rate: {visual_lr}")
    main_print(f"   - Text learning rate: {text_lr}")
    main_print(f"   - Logit scale learning rate: {logit_scale_lr}")

    backbone_wd = 0.0 if args.use_l2_reg else args.weight_decay
    optimizer_grouped_parameters = [
        # Vision Model parameters
        {
            "params": [p for n, p in model.named_parameters() if "vision_model." in n and p.requires_grad],
            "lr": visual_lr,
            "weight_decay": backbone_wd
        },
        # Text Model parameters
        {
            "params": [p for n, p in model.named_parameters() if "text_model." in n and p.requires_grad],
            "lr": text_lr,
            "weight_decay": backbone_wd
        },
        # Logit Scale parameter
        {
            "params": [p for n, p in model.named_parameters() if "logit_scale" in n and p.requires_grad],
            "lr": logit_scale_lr,
            "weight_decay": 0.0 
        },
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
    
    # 确保所有参数都被分配
    assigned_params = set()
    for group in optimizer_grouped_parameters:
        assigned_params.update([p for p in group["params"]])
    
    all_params = set(p for p in model.parameters() if p.requires_grad)
    
    if all_params != assigned_params:
         unassigned_params = all_params - assigned_params
         unassigned_names = [n for n, p in model.named_parameters() if p in unassigned_params]
         main_print(f"!! WARNING !!: Some parameters were not assigned a learning rate group: {unassigned_names}")
         default_group = {
             "params": list(unassigned_params),
             "lr": default_lr,
         }
         optimizer_grouped_parameters.append(default_group)

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=default_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        model_type=model_type,
        use_sigmoid_loss=args.use_sigmoid_loss,
        use_l2_reg=args.use_l2_reg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[BestModelCallback()],
        optimizers=(optimizer, None),
        orig_model=orig_model,
    )

    if args.use_l2_reg and orig_model is not None:
        trainer.orig_model = orig_model.to(accelerator.device)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    accelerator.wait_for_everyone()

    best_model_path = args.best_model_dir
    if accelerator.is_main_process:
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        main_print(f"\n💾 Best model (eval_loss) saved to: {best_model_path}")

    accelerator.wait_for_everyone()
    return best_model_path


def push_to_hub(best_model_path, repo_name, model_type="clip"):
    if accelerator.is_main_process:
        main_print(f"\n🤗 Pushing model to HuggingFace Hub: {repo_name}")
        
        if model_type == "clip":
            model = CLIPModel.from_pretrained(best_model_path)
            processor = CLIPProcessor.from_pretrained(best_model_path)
        elif model_type == "siglip":
            model = SiglipModel.from_pretrained(best_model_path)
            processor = SiglipProcessor.from_pretrained(best_model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.push_to_hub(repo_name)
        processor.push_to_hub(repo_name)

        main_print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    main_print(f"Arguments: {args}")
    
    best_model_path = train_clip(args)
    
    if args.push_to_hub:
        push_to_hub(best_model_path, args.hub_model_name, args.model_type)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        main_print("\n🎉 All processes have completed successfully!")


if __name__ == "__main__":
    main()