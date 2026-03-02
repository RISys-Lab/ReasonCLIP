import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel

# Make script runnable without manually exporting PYTHONPATH.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import rank0_print


def _load_non_lora_trainables(model, model_path):
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        raise FileNotFoundError(f"Missing non_lora_trainables.bin in {model_path}")

    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)


def _merge_qwen_lora(model_path, model_base):
    cfg = AutoConfig.from_pretrained(model_path)
    model_type = getattr(cfg, "model_type", "")
    rank0_print(f"Merging Qwen-family LoRA, model_type={model_type}")

    # LoRA checkpoints may not always contain full tokenizer files.
    # Prefer model_path, fallback to base model tokenizer.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception as e:
        rank0_print(f"Tokenizer not found in model_path, fallback to model_base. reason: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    if model_type == "qwen3":
        from llava.model.language_model.llava_qwen3 import LlavaQwen3Config, LlavaQwen3ForCausalLM

        llava_cfg = LlavaQwen3Config.from_pretrained(model_path)
        model = LlavaQwen3ForCausalLM.from_pretrained(
            model_base,
            low_cpu_mem_usage=True,
            config=llava_cfg,
            attn_implementation="sdpa",
        )
    else:
        from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

        llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
        model = LlavaQwenForCausalLM.from_pretrained(
            model_base,
            low_cpu_mem_usage=True,
            config=llava_cfg,
            attn_implementation="sdpa",
        )

    _load_non_lora_trainables(model, model_path)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    return tokenizer, model


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    cfg = AutoConfig.from_pretrained(args.model_path)
    model_type = getattr(cfg, "model_type", "")

    # Explicit branch for Qwen-family LoRA to avoid ambiguous generic loader behavior.
    if "qwen" in model_type or "qwen" in model_name.lower():
        tokenizer, model = _merge_qwen_lora(args.model_path, args.model_base)
    else:
        tokenizer, model, _image_processor, _context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, device_map="cpu"
        )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    rank0_print(f"Merged model saved to {args.save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
