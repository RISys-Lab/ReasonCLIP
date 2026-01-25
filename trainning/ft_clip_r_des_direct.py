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
from torch.autograd import Function
from accelerate import Accelerator
import numpy as np 
from typing import Optional, List
import math
import glob
# from lion_pytorch import Lion
from torch.optim import AdamW
import torch.distributed as dist
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
    # print("Wandb disabled for non-main processes")  # 这个保留，让每个进程都知道自己的状态

import wandb
import argparse
from transformers import TrainerCallback
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning and uploading CLIP model to HuggingFace Hub")
    # Training parameters
    
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

    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (Ampere+ GPUs)")

    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed JSON config (ZeRO, etc.)")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use FlashAttention-2 backend for attention (attn_implementation=flash_attention_2).",
    )
    
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
    
    
    # Hub push parameters
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Whether to push to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, default="fesvhtr", 
                        help="HuggingFace username")
    parser.add_argument("--hub_model_name", type=str, default="clip-iferniu-L14-10epoch", 
                        help="Model name on the Hub")
    
    # Dataset parameters
    parser.add_argument("--parquet_files_ReasonLite", type=str, nargs="+", required=True,
                        help="Paths to one or more parquet files (space-separated or glob) for ReasonLite")
    parser.add_argument("--parquet_files_ReasonPro", type=str, nargs="+", required=True,
                        help="Paths to one or more parquet files (space-separated or glob) for ReasonPro")
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
    def __init__(self, model_type="clip",orig_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type  # "clip" 或 "siglip
        self.backbone = None
    # @staticmethod
    # def _siglip_logistic_loss(
    #     logits: torch.Tensor, 
    #     labels: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     SigLIP 风格的 logistic loss:
    #     logits: [B, N]，N = world_size * B
    #     labels: [B]，每一行正样本所在的列索引（全局索引）
    #     """
    #     B = logits.size(0)
    #     device = logits.device

    #     # 构造 +1 / -1 的 label matrix
    #     label_matrix = logits.new_full(logits.shape, -1.0)   # 全部初始化为 -1
    #     row_idx = torch.arange(B, device=device)
    #     label_matrix[row_idx, labels] = 1.0                  # 正样本位置设为 +1

    #     # 对所有 pair 做 -log σ(z_ij * logit_ij) 的平均
    #     loss = -F.logsigmoid(label_matrix * logits).mean()
    #     return loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 动态获取 backbone（此时 model 已经被 DDP 包装）
        if self.backbone is None:
            self.backbone = model.module if hasattr(model, "module") else model
        
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
        text_features = self.backbone.get_text_features(
            input_ids=inputs["text_input_ids"],
            attention_mask=inputs["text_attention_mask"],
        )
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # ---- temperature (clamp to avoid blow-up in large-batch) ----
        with torch.no_grad():
            # 与 CLIP/SigLIP 通用：限制 logit_scale 的上界
            if hasattr(self.backbone, "logit_scale"):
                self.backbone.logit_scale.data.clamp_(max=math.log(100.0))
        logit_scale = self.backbone.logit_scale.exp() if hasattr(self.backbone, "logit_scale") else 1.0

        logit_bias = getattr(self.backbone, "logit_bias", None)
        if logit_bias is not None:
            bias = logit_bias.to(image_features.dtype)
        else:
            bias = 0.0

        # ---- cross-GPU gather (only local slice keeps grad) ----
        B = image_features.size(0)
        rank = accelerator.process_index
        all_image = gather_with_local_grad(image_features, accelerator)   # [world*B, D]
        all_trp   = gather_with_local_grad(text_features, accelerator)
        # all_image = all_gather_with_local_grad(image_features)
        # all_trp = all_gather_with_local_grad(trp_text_features)

        # global labels: shift by the local slice offset
        labels_global = torch.arange(B, device=device) + rank * B  # [B]


        # ---- TRP branch ----
        trp_logits_per_image = logit_scale * (image_features     @ all_trp.t()) + bias
        trp_logits_per_text  = logit_scale * (text_features  @ all_image.t()) + bias

        
        trp_loss = 0.5 * (
            F.cross_entropy(trp_logits_per_image, labels_global) +
            F.cross_entropy(trp_logits_per_text,  labels_global)
        )

        del trp_logits_per_image, trp_logits_per_text

        # ---- combine ----
        contrastive_loss = trp_loss
        total_loss = contrastive_loss

        # ---- optional logging ----
        if accelerator.is_main_process and "wandb" in self.args.report_to:
            import wandb
            wandb.log(
                {
                    "train/contrastive_loss": contrastive_loss.item(),
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
            loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, None)

class CLIPRLiteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, processor):
        self.dataset = dataset_dict
        self.processor = processor
        # 每个样本有3个TRL caption，每个caption生成一个训练样本
        self.captions_per_image = 3
        proc_name = processor.__class__.__name__.lower()
        if "siglip" in proc_name:
            self.text_max_len = 64
        else:
            self.text_max_len = 77
    
    def __len__(self):
        # 每个原始样本生成3个训练样本（对应3个TRL caption）
        return len(self.dataset) * self.captions_per_image
    
    def __getitem__(self, idx):
        # 计算原始样本索引和TRL caption索引
        original_idx = idx // self.captions_per_image
        trl_idx = idx % self.captions_per_image
        item = self.dataset[original_idx]
        
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # 只使用TRL caption，不使用TB
        trl_captions = item["trl"]
        trl_caption = trl_captions[trl_idx]  # 选择对应的TRL caption
        
        img_enc = self.processor(images=image, return_tensors="pt")
        trl_enc = self.processor(text=[trl_caption], return_tensors="pt", padding="max_length", truncation=True, max_length=self.text_max_len)
        
        # 构建返回的batch，只包含图像和TRL文本
        return {
            "pixel_values": img_enc["pixel_values"].squeeze(0),
            "text_input_ids": trl_enc["input_ids"].squeeze(0),
            "text_attention_mask": trl_enc.get("attention_mask", torch.ones_like(trl_enc["input_ids"])).squeeze(0),
        }

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


def train_clip(args):
    # 获取当前是否为主进程
    is_main_process = accelerator.is_main_process

    # 先取出模型信息，避免后面打印时 NameError
    model_name = args.model_name
    model_type = args.model_type

    # 打印分布式训练信息
    main_print("="*60)
    main_print("🚀 CLIP-R Training Configuration")
    main_print("="*60)
    main_print(f"🔧 Training setup:")
    main_print(f"   - Model type: {model_type.upper()}")
    main_print(f"   - Model name: {model_name}")
    main_print(f"   - Distributed training: {accelerator.num_processes > 1}")
    main_print(f"   - Number of processes: {accelerator.num_processes}")
    main_print(f"   - Mixed precision: {accelerator.mixed_precision}")
    if accelerator.num_processes > 1:
        main_print(f"   - Current process rank: {accelerator.process_index}")

    # 统一 run_id：仅主进程生成，然后广播给所有进程
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

    # 只主进程创建目录，其它进程等待
    if accelerator.is_main_process:
        os.makedirs(os.path.join(save_root, "finetune_weights"), exist_ok=True)
        os.makedirs(os.path.join(save_root, "best_model"), exist_ok=True)
    accelerator.wait_for_everyone()

    # 统一 run 名称和保存路径
    args.run_name = f"{args.run_name}_{run_id.replace('run_', '')}"
    args.output_dir = os.path.join(save_root, "finetune_weights")
    args.best_model_dir = os.path.join(save_root, "best_model")

    # 初始化 WandB（仅主进程；其余进程禁用）
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
        # processor_name = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/clip-vit-large-patch14-336"
        processor = CLIPProcessor.from_pretrained(model_name)
    elif model_type == "siglip":
        model = SiglipModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        )
        # 加载原始模型用于L2正则化
        orig_model = SiglipModel.from_pretrained(model_name)
        for p in orig_model.parameters():
            p.requires_grad = False
        # processor_name = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip2-so400m-patch14-384"
        processor = SiglipProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    # ================================ 数据集配置 ================================
    # 读取parquet数据集
    # 替换以下导入处附近：去掉 pandas 的读取用法
    from datasets import load_dataset

    # ================================ 数据集配置（替换整段 pandas 读取与切分） ================================
    main_print(f"\n📊 Dataset Configuration:")
    main_print(f"   - Loading from: {args.parquet_files_ReasonLite}")
    main_print(f"   - Loading from: {args.parquet_files_ReasonPro}")

    files_ReasonLite = sorted(args.parquet_files_ReasonLite)
    files_ReasonPro = sorted(args.parquet_files_ReasonPro)
    main_print(f"📊 Loading {len(files_ReasonLite)} parquet files...")
    main_print(f"📊 Loading {len(files_ReasonPro)} parquet files...")
    hf_ds_ReasonLite = load_dataset(
        "parquet",
        data_files={"train": files_ReasonLite},   # 用 dict 明确 split
        split="train",
        keep_in_memory=False
    )
    hf_ds_ReasonPro = load_dataset(
        "parquet",
        data_files={"train": files_ReasonPro},   # 用 dict 明确 split
        split="train",
        keep_in_memory=False
    )
    main_print(f"   - ReasonLite samples: {len(hf_ds_ReasonLite)}")
    main_print(f"   - ReasonPro samples: {len(hf_ds_ReasonPro)}")

    # 2) 分别切分两个数据集（用 HF 自带的 split）
    if args.use_split:
        # ReasonLite 切分
        split1_lite = hf_ds_ReasonLite.train_test_split(test_size=0.2, seed=42)
        train_lite_hf = split1_lite["train"]
        tmp_lite = split1_lite["test"].train_test_split(test_size=0.5, seed=42)
        eval_lite_hf, test_lite_hf = tmp_lite["train"], tmp_lite["test"]
        
        # ReasonPro 切分
        split1_pro = hf_ds_ReasonPro.train_test_split(test_size=0.2, seed=42)
        train_pro_hf = split1_pro["train"]
        tmp_pro = split1_pro["test"].train_test_split(test_size=0.5, seed=42)
        eval_pro_hf, test_pro_hf = tmp_pro["train"], tmp_pro["test"]
        
        main_print(f"   - ReasonLite split: {len(train_lite_hf)} train, {len(eval_lite_hf)} eval, {len(test_lite_hf)} test")
        main_print(f"   - ReasonPro split: {len(train_pro_hf)} train, {len(eval_pro_hf)} eval, {len(test_pro_hf)} test")
    else:
        if args.holdout_ratio > 0:
            # ReasonLite 切分
            split_lite = hf_ds_ReasonLite.train_test_split(test_size=args.holdout_ratio, seed=42)
            train_lite_hf, eval_lite_hf = split_lite["train"], split_lite["test"]
            
            # ReasonPro 切分
            split_pro = hf_ds_ReasonPro.train_test_split(test_size=args.holdout_ratio, seed=42)
            train_pro_hf, eval_pro_hf = split_pro["train"], split_pro["test"]
            
            main_print(f"   - ReasonLite holdout: {len(train_lite_hf)} train, {len(eval_lite_hf)} eval")
            main_print(f"   - ReasonPro holdout: {len(train_pro_hf)} train, {len(eval_pro_hf)} eval")
        else:
            train_lite_hf, eval_lite_hf = hf_ds_ReasonLite, None
            train_pro_hf, eval_pro_hf = hf_ds_ReasonPro, None
            main_print(f"   - No eval holdout")

    # 3) 用不同的自定义 Dataset 类处理不同的数据格式
    # ReasonLite 用 CLIPRLiteDataset（每个样本扩展成3个，因为有3个TRL caption）
    train_lite_dataset = CLIPRLiteDataset(train_lite_hf, processor)
    eval_lite_dataset = CLIPRLiteDataset(eval_lite_hf, processor) if eval_lite_hf else None
    
    # ReasonPro 用 CLIPRProDataset（1:1映射，每个样本1个TRP caption）
    train_pro_dataset = CLIPRProDataset(train_pro_hf, processor)
    eval_pro_dataset = CLIPRProDataset(eval_pro_hf, processor) if eval_pro_hf else None
    
    # 4) 合并处理后的数据集（使用 PyTorch 的 ConcatDataset）
    # 处理后的格式已经统一：都是 (image, text_input_ids, text_attention_mask)
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_lite_dataset, train_pro_dataset])
    eval_dataset = ConcatDataset([eval_lite_dataset, eval_pro_dataset]) if (eval_lite_hf is not None or eval_pro_hf is not None) else None
    
    main_print(f"   - Train Lite dataset size: {len(train_lite_dataset)}")
    main_print(f"   - Train Pro dataset size: {len(train_pro_dataset)}")
    main_print(f"   - Combined train dataset size: {len(train_dataset)}")
    
    main_print(f"   - Train Lite dataset size: {len(train_lite_dataset)}")
    main_print(f"   - Train Pro dataset size: {len(train_pro_dataset)}")
    main_print(f"   - Combined train dataset size: {len(train_dataset)}")

    main_print(f"   - Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        main_print(f"   - Eval dataset size: {len(eval_dataset)}")
    
    
    # 验证数据样本
    main_print(f"\n🔍 Data Validation:")
    sample = train_dataset[0]
    main_print(f"   - Sample keys: {list(sample.keys())}")
    main_print(f"   - Text Input IDs shape: {sample['text_input_ids'].shape}")
    main_print(f"   - Text Attention Mask shape: {sample['text_attention_mask'].shape}")
    main_print(f"   - Pixel values shape: {sample['pixel_values'].shape}")
    main_print(f"   - ✅ Data format validated: (image, trl_text) - 3 TRL captions per image")
    
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
        save_total_limit=args.save_total_limit,  # 保留更多检查点
        report_to="wandb" if args.wandb_log and is_main_process else "none",
        run_name=args.run_name,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # 使用验证损失作为指标
        greater_is_better=False,       # 损失越小越好
        dataloader_num_workers=args.num_workers,           # 默认: 0 (主进程加载)
        dataloader_pin_memory=True,         # 默认: True
        remove_unused_columns=False,
        # 分布式训练配置
        ddp_find_unused_parameters=False,  # 关闭unused parameters检测，提高性能
        dataloader_drop_last=True,            # 丢弃训练集最后一个不满 batch
        deepspeed=args.deepspeed,
        seed=42,
        data_seed=42,
    )
    
    
    main_print(f"\n🎯 Loss Configuration:")
    main_print(f"   - Contrastive loss (TRL): 1.0")
    
    # ================================ 断点恢复配置 ================================
    resume_from_checkpoint = None
    
    if args.resume_from_checkpoint:
        # 用户指定了具体的checkpoint路径
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
    # 推荐的学习率: backbone 使用较低的 LR，logit_scale 使用主 LR
    default_lr = args.default_lr
    visual_lr = args.visual_lr
    text_lr = args.text_lr
    logit_scale_lr = args.logit_scale_lr
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
            "weight_decay": 0.0 # 通常不对 logit_scale 应用 weight decay
        },
    ]

    # 过滤掉没有参数的组 (以防万一)
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
         # 可以选择将它们添加到默认组，或报错
         # 这里简单地将它们添加到默认组 (default_lr)
         default_group = {
             "params": list(unassigned_params),
             "lr": default_lr,
         }
         optimizer_grouped_parameters.append(default_group)

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=default_lr, # 默认 lr (虽然每个组都有指定)
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999), # AdamW 默认 betas
        eps=1e-8          # AdamW 默认 epsilon
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
    )

    # 将orig_model移动到设备上用于L2正则化
    trainer.orig_model = orig_model.to(accelerator.device)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # trainer.train() 结束后, trainer.model 已经是最佳模型了
    accelerator.wait_for_everyone() # 等待所有进程完成

    best_model_path = args.best_model_dir
    
    # 只有主进程保存
    if accelerator.is_main_process:
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        main_print(f"\n💾 Best model (eval_loss) saved to: {best_model_path}")
        if trainer.state.best_model_checkpoint:
            main_print(f"   (Best checkpoint was: {trainer.state.best_model_checkpoint})")

    accelerator.wait_for_everyone() # 确保保存后再继续
 
    return best_model_path


def push_to_hub(best_model_path, repo_name, model_type="clip"):
    # 只有主进程执行推送操作
    if accelerator.is_main_process:
        main_print(f"\n🤗 Pushing model to HuggingFace Hub: {repo_name}")
        from huggingface_hub import HfApi
        
        # 根据模型类型加载相应的模型和处理器
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
    # No need to print skip message for non-main processes

    # 等待所有进程同步
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    main_print(f"Arguments: {args}")
    
    best_model_path = train_clip(args)
    
    # 推送到HuggingFace Hub
    if args.push_to_hub:
        push_to_hub(best_model_path, args.hub_model_name, args.model_type)
    
    # Distributed training cleanup
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        main_print("\n🎉 All processes have completed successfully!")


if __name__ == "__main__":
    main()
