import os
from PIL import Image
import torch
from datasets import Dataset, load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    Siglip2Processor,
    Siglip2Model,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from accelerate import Accelerator
import numpy as np 
from typing import Optional, List
import math
import glob
from torch.optim import AdamW
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

import wandb
import argparse
from transformers import TrainerCallback
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    # =========================================================================
    # 注意：为了兼容现有 .sh 脚本，保留了所有参数定义，
    # 即使某些参数（如 classifier_lr, gamma_adv）在下方逻辑中不再被使用。
    # =========================================================================
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "siglip"],
                        help="Model type: 'clip' or 'siglip'")
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
    parser.add_argument("--default_lr", type=float, default=1e-4, 
                            help="Learning rate")
    parser.add_argument("--visual_lr", type=float, default=None, 
                            help="Learning rate")
    parser.add_argument("--text_lr", type=float, default=None, 
                            help="Learning rate")
    parser.add_argument("--logit_scale_lr", type=float, default=None, 
                            help="Learning rate")
    parser.add_argument("--classifier_lr", type=float, default=None, 
                            help="Learning rate (UNUSED in this pure finetune version)")

    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (Ampere+ GPUs)")

    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed JSON config (ZeRO, etc.)")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use FlashAttention-2 backend for attention (attn_implementation=flash_attention_2).",
    )
    
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
    
    # Evaluation parameters
    parser.add_argument("--eval_strategy", type=str, default="ratio", choices=["steps", "epoch", "ratio"],
                        help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=250, 
                        help="Evaluation steps")
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Evaluation ratio")
    
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
    
    # Hub push parameters
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="fesvhtr", 
                        help="HuggingFace username")
    parser.add_argument("--hub_model_name", type=str, default="clip-iferniu-L14-10epoch", 
                        help="Model name on the Hub")
    
    # Dataset parameters
    parser.add_argument("--parquet_files", type=str, nargs="+", required=True,
                        help="Paths to one or more parquet files (space-separated or glob)")
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
                        help="wandb entity name (team or username)")
    parser.add_argument("--wandb_log", action="store_true",
                        help="Enable wandb logging")
    
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")

    parser.add_argument("--gamma_adv", type=float, default=0.1,
                        help="Gamma for adversarial classification loss (UNUSED in this version)")
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

# ReasoningClassifier 已删除

class CLIPTrainer(Trainer):
    def __init__(self, model_type="clip", orig_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type  # "clip" 或 "siglip"
        self.orig_model = orig_model  # 允许外部传进来
        self.orig_state = None
        self.backbone = None  # 延迟初始化
    
    def _initialize_l2_reg(self, model=None):
        """在模型移动到设备后，安全地初始化原始权重状态"""
        if self.orig_state is None and self.orig_model is not None:
            # 获取模型所在设备（兼容 DDP 包装的模型）
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
        label_matrix = logits.new_full(logits.shape, -1.0)   # 全部初始化为 -1
        row_idx = torch.arange(B, device=device)
        label_matrix[row_idx, labels] = 1.0                  # 正样本位置设为 +1

        # 对所有 pair 做 -log σ(z_ij * logit_ij) 的平均
        loss = -F.logsigmoid(label_matrix * logits).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, use_sigmoid_loss=False):
        # 动态获取 backbone（此时 model 已经被 DDP 包装）
        if self.backbone is None:
            self.backbone = model.module if hasattr(model, "module") else model
        
        if self.orig_model is not None and self.orig_state is None:
            self._initialize_l2_reg(model)

        # ---- helper: gather across GPUs but keep local slice with gradient ----
        def gather_with_local_grad(x, accelerator):
            if accelerator.num_processes == 1:
                return x
            B = x.size(0)
            gathered = accelerator.gather(x.detach())  # [world*B, D], no grad
            parts = []
            for i in range(accelerator.num_processes):
                if i == accelerator.process_index:
                    parts.append(x)  # keep grad for local slice
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
            # 与 CLIP/SigLIP 通用：限制 logit_scale 的上界
            if hasattr(self.backbone, "logit_scale"):
                self.backbone.logit_scale.data.clamp_(max=math.log(50.0))
        logit_scale = self.backbone.logit_scale.exp() if hasattr(self.backbone, "logit_scale") else 1.0

        if use_sigmoid_loss and self.model_type == "siglip":
            logit_bias = getattr(self.backbone, "logit_bias", None)
            bias = logit_bias.to(image_features.dtype) if logit_bias is not None else 0.0
        else:
            bias = 0.0  # CLIP 模式强制无 Bias

        # ---- cross-GPU gather (only local slice keeps grad) ----
        B = image_features.size(0)
        rank = accelerator.process_index
        all_image = gather_with_local_grad(image_features, accelerator)   # [world*B, D]
        all_trp   = gather_with_local_grad(trp_text_features, accelerator)

        # global labels: shift by the local slice offset
        labels_global = torch.arange(B, device=device) + rank * B  # [B]

        # ---- TRP branch (Main Contrastive Loss) ----
        trp_logits_per_image = logit_scale * (image_features     @ all_trp.t()) + bias
        trp_logits_per_text  = logit_scale * (trp_text_features  @ all_image.t()) + bias

        if use_sigmoid_loss and self.model_type == "siglip":
            trp_loss = self._siglip_logistic_loss(trp_logits_per_image, labels_global)
        else:
            # we also use cross entropy loss for siglip
            trp_loss = 0.5 * (
                F.cross_entropy(trp_logits_per_image, labels_global) +
                F.cross_entropy(trp_logits_per_text,  labels_global)
            )

        del trp_logits_per_image, trp_logits_per_text

        # ---- combine ----
        total_loss = trp_loss

        # ---- L2 Regularization (Keep orig weights constraint) ----
        if self.orig_state is not None:
            beta = 1e-5 # 这是 L2 权重
            l2_reg = torch.zeros((), device=device, dtype=torch.float32)
            
            for name, p in self.backbone.named_parameters():
                if ("vision_model." in name) or ("text_model." in name):
                    if ("projection" in name) or ("logit_scale" in name):
                        continue
                    if name in self.orig_state:
                        p0 = self.orig_state[name]
                        # 确保 p0 和 p 在同一设备
                        l2_reg = l2_reg + (p.float() - p0.to(p.device)).pow(2).sum()
            total_loss = total_loss + beta * l2_reg
        
        # [REMOVED] Adversarial classification loss
        # total_loss = total_loss + self.gamma_adv * (loss_cls_text + loss_cls_image)

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

        # ---- optional outputs (detached; global view) ----
        if return_outputs:
            with torch.no_grad():
                trp_logits_view = (image_features @ all_trp.t()).detach()
            outputs = {
                "trp_image_text_logits": trp_logits_view,
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
        model.eval()
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, None)


class CLIPRDataset(torch.utils.data.Dataset):
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
        # [REMOVED] trp_cls handling
        # trp_cls = item["trp_cls"]
        # trp_cls_idx = trp_cls_to_idx[trp_cls]

        img_enc = self.processor(images=image, return_tensors="pt")
        trp_enc = self.processor(text=trp_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.text_max_len)
        
        return {
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "trp_input_ids": trp_enc["input_ids"].squeeze(0),
            "trp_attention_mask": trp_enc.get("attention_mask", torch.ones_like(trp_enc["input_ids"])).squeeze(0),
            # [REMOVED] "trp_cls": trp_cls_idx, 
        }


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = accelerator.is_main_process

    # 先取出模型信息，避免后面打印时 NameError
    model_name = args.model_name
    model_type = args.model_type

    # 打印分布式训练信息
    main_print("="*60)
    main_print("🚀 CLIP-R Training Configuration (PURE FINETUNE)")
    main_print("="*60)
    main_print(f"🔧 Training setup:")
    main_print(f"   - Model type: {model_type.upper()}")
    main_print(f"   - Model name: {model_name}")
    main_print(f"   - Distributed training: {accelerator.num_processes > 1}")
    main_print(f"   - Number of processes: {accelerator.num_processes}")
    main_print(f"   - Mixed precision: {accelerator.mixed_precision}")
    if accelerator.num_processes > 1:
        main_print(f"   - Current process rank: {accelerator.process_index}")

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

    # 所有进程都使用同一个 save_root
    save_root = os.path.join(args.output_dir, run_id)

    # 只主进程创建目录
    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_root, "finetune_weights"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "best_model"), exist_ok=True)
    accelerator.wait_for_everyone()

    # 统一 run 名称和保存路径
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
    
    # 根据模型类型加载相应的模型和处理器
    if model_type == "clip":
        model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        # 加载原始模型用于L2正则化
        orig_model = CLIPModel.from_pretrained(model_name)
        for p in orig_model.parameters():
            p.requires_grad = False
        processor_name = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(processor_name)
    elif model_type == "siglip":
        model = Siglip2Model.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        # 加载原始模型用于L2正则化
        orig_model = Siglip2Model.from_pretrained(model_name)
        for p in orig_model.parameters():
            p.requires_grad = False
        processor_name = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
        processor = Siglip2Processor.from_pretrained(processor_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # [REMOVED] Classifier initialization block
    # main_print(f"🔧 Initializing Reasoning Classifier...")
    # backbone = model.module if hasattr(model, "module") else model
    # ...

    # ================================ 数据集配置 ================================
    main_print(f"\n📊 Dataset Configuration:")
    main_print(f"   - Loading from: {args.parquet_files}")

    files = sorted(args.parquet_files)
    main_print(f"📊 Loading {len(files)} parquet files...")
    hf_ds = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
        keep_in_memory=False
    )
    main_print(f"   - Total samples (rows): {len(hf_ds)}")

    # 切分
    if args.use_split:
        split1 = hf_ds.train_test_split(test_size=0.2, seed=42)          # 8:2
        train_hf = split1["train"]
        tmp     = split1["test"].train_test_split(test_size=0.5, seed=42) # 2 -> 1:1
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

    # 构建自定义 Dataset
    train_dataset = CLIPRDataset(train_hf, processor)
    eval_dataset  = CLIPRDataset(eval_hf, processor) if eval_hf else None

    main_print(f"   - Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        main_print(f"   - Eval dataset size: {len(eval_dataset)}")
    
    # 验证数据样本
    main_print(f"\n🔍 Data Validation:")
    sample = train_dataset[0]
    main_print(f"   - Sample keys: {list(sample.keys())}")
    main_print(f"   - TRP Input IDs shape: {sample['trp_input_ids'].shape}")
    main_print(f"   - TRP Attention Mask shape: {sample['trp_attention_mask'].shape}")
    main_print(f"   - Pixel values shape: {sample['pixel_values'].shape}")
    
    # ================================ 训练参数配置 ================================
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
    
    logging_steps, logging_strategy = compute_strategy_steps(
        args.logging_strategy, args.logging_ratio, args.logging_steps, total_steps, steps_per_epoch)
    save_steps, save_strategy = compute_strategy_steps(
        args.save_strategy, args.save_ratio, args.save_steps, total_steps, steps_per_epoch)
    eval_steps, eval_strategy = compute_strategy_steps(
        args.eval_strategy, args.eval_ratio, args.eval_steps, total_steps, steps_per_epoch)
    
    main_print(f"\n📝 Logging & Evaluation Schedule:")
    main_print(f"   - Logging: every {logging_steps} steps ({args.logging_strategy})")
    main_print(f"   - Saving: every {save_steps} steps ({args.save_strategy})")
    main_print(f"   - Evaluation: every {eval_steps} steps ({args.eval_strategy})")

    # 训练参数
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
    main_print(f"   - Adversarial classification: DISABLED")
    
    # ================================ 断点恢复配置 ================================
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            resume_from_checkpoint = args.resume_from_checkpoint
            main_print(f"\n🔄 Resuming from specified checkpoint: {resume_from_checkpoint}")
        else:
            main_print(f"❌ Specified checkpoint not found: {args.resume_from_checkpoint}")
            main_print("   Starting training from scratch...")

    if resume_from_checkpoint:
        base = os.path.basename(os.path.normpath(resume_from_checkpoint))
        step_number = base.split("checkpoint-")[-1] if base.startswith("checkpoint-") else "unknown"
        main_print(f"   - Resuming from step: {step_number}")


    main_print(f"\n🚀 Starting Training...")
    main_print("="*60)
    
    main_print(f"🔧 Setting up optimizer with different learning rates...")
    default_lr = args.default_lr
    visual_lr = args.visual_lr
    text_lr = args.text_lr
    logit_scale_lr = args.logit_scale_lr
    # classifier_lr = args.classifier_lr # Ignored

    main_print(f"   - Default learning rate: {default_lr}")
    main_print(f"   - Visual learning rate: {visual_lr}")
    main_print(f"   - Text learning rate: {text_lr}")
    main_print(f"   - Logit scale learning rate: {logit_scale_lr}")

    optimizer_grouped_parameters = [
        # Vision Model parameters
        {
            "params": [p for n, p in model.named_parameters() if "vision_model." in n and p.requires_grad],
            "lr": visual_lr,
        },
        # Text Model parameters
        {
            "params": [p for n, p in model.named_parameters() if "text_model." in n and p.requires_grad],
            "lr": text_lr,
        },
        # Logit Scale parameter
        {
            "params": [p for n, p in model.named_parameters() if "logit_scale" in n and p.requires_grad],
            "lr": logit_scale_lr,
            "weight_decay": 0.0
        },
        # [REMOVED] Classifier parameter groups
    ]

    # 过滤掉没有参数的组
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
    
    # 确保所有参数都被分配了
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
    main_print(f"   - Optimizer: AdamW")
    main_print(f"   - Default weight decay: {args.weight_decay}")

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        model_type=model_type,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[BestModelCallback()],
        optimizers=(optimizer, None),
        orig_model=orig_model,
        # Removed gamma_adv and num_classes
    )

    trainer.orig_model = orig_model.to(accelerator.device)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    accelerator.wait_for_everyone()

    best_model_path = args.best_model_dir
    
    if accelerator.is_main_process:
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        main_print(f"\n💾 Best model (eval_loss) saved to: {best_model_path}")
        if trainer.state.best_model_checkpoint:
            main_print(f"   (Best checkpoint was: {trainer.state.best_model_checkpoint})")

    accelerator.wait_for_everyone()
 
    return best_model_path


def push_to_hub(best_model_path, repo_name, model_type="clip"):
    if accelerator.is_main_process:
        main_print(f"\n🤗 Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        if model_type == "clip":
            model = CLIPModel.from_pretrained(best_model_path)
            processor = CLIPProcessor.from_pretrained(best_model_path)
        elif model_type == "siglip":
            model = Siglip2Model.from_pretrained(best_model_path)
            processor = Siglip2Processor.from_pretrained(best_model_path)
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