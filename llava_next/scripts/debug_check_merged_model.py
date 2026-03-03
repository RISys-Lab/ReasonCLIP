#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
import copy

import torch


def check_mm_projector_keys(model_path: str) -> bool:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"[ERROR] Missing index file: {index_path}")
        return False

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    weight_map = idx.get("weight_map", {})
    all_keys = list(weight_map.keys())
    mm_projector_keys = [k for k in all_keys if "mm_projector" in k]
    vision_tower_keys = [k for k in all_keys if "vision_tower" in k]
    vision_resampler_keys = [k for k in all_keys if "vision_resampler" in k]

    print("=== Check 1: multimodal weights in merged model ===")
    print(f"total param keys in index: {len(all_keys)}")
    print(f"mm_projector keys: {len(mm_projector_keys)}")
    print(f"vision_tower keys: {len(vision_tower_keys)}")
    print(f"vision_resampler keys: {len(vision_resampler_keys)}")
    if mm_projector_keys:
        print("sample mm_projector keys:")
        for k in mm_projector_keys[:8]:
            print(f"  - {k}")
    else:
        print("[WARN] No mm_projector keys found in merged weights.")

    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"config.use_mm_proj: {cfg.get('use_mm_proj', None)}")
        print(f"config.mm_vision_tower: {cfg.get('mm_vision_tower', None)}")
        print(f"config.model_type: {cfg.get('model_type', None)}")

    return len(mm_projector_keys) > 0


def run_text_only_sanity(
    model_path: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    print("\n=== Check 2: text-only generation sanity ===")
    from llava.model.builder import load_pretrained_model

    tokenizer, model, _image_processor, _max_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    model.eval()

    # Get a real device from model parameters for safe tensor placement.
    model_device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    gen_ids = out[:, input_ids.shape[1] :]
    print(f"first 20 generated token ids: {gen_ids[0, :20].tolist()}")
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print(f"decoded text: {text}")

    if gen_ids.numel() > 0:
        unique_ids = torch.unique(gen_ids).numel()
        print(f"generated unique token count: {int(unique_ids)}")


def _print_last_token_stats(tag: str, logits: torch.Tensor, tokenizer) -> None:
    last = logits[0, -1].float()
    is_finite = torch.isfinite(last)
    finite_ratio = float(is_finite.float().mean().item())
    print(f"[{tag}] last-step logits finite ratio: {finite_ratio:.6f}")
    if finite_ratio == 0.0:
        print(f"[{tag}] all logits are non-finite (NaN/Inf).")
        return

    safe_last = torch.where(is_finite, last, torch.full_like(last, -1e30))
    topk_vals, topk_ids = torch.topk(safe_last, k=5, dim=-1)
    print(f"[{tag}] top1 token id: {int(topk_ids[0].item())}, token: {repr(tokenizer.decode([int(topk_ids[0].item())]))}")
    print(f"[{tag}] top5:")
    for i in range(5):
        tid = int(topk_ids[i].item())
        tval = float(topk_vals[i].item())
        tstr = tokenizer.decode([tid])
        print(f"  - id={tid:<8d} logit={tval:>10.4f} token={repr(tstr)}")


def run_parameter_finite_check(
    model_path: str,
    model_name: str,
) -> None:
    print("\n=== Check 4: parameter finite check ===")
    from llava.model.builder import load_pretrained_model

    _tokenizer, model, _image_processor, _max_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    model.eval()

    groups = {
        "vision_tower": ("vision_tower",),
        "mm_projector": ("mm_projector",),
        "vision_resampler": ("vision_resampler",),
    }

    for group_name, keywords in groups.items():
        total_tensors = 0
        bad_tensors = 0
        bad_first = None
        bad_first_non_finite = 0
        for n, p in model.named_parameters():
            if not any(k in n for k in keywords):
                continue
            total_tensors += 1
            finite_mask = torch.isfinite(p.detach())
            non_finite_cnt = int((~finite_mask).sum().item())
            if non_finite_cnt > 0:
                bad_tensors += 1
                if bad_first is None:
                    bad_first = n
                    bad_first_non_finite = non_finite_cnt

        print(
            f"[{group_name}] tensors={total_tensors}, "
            f"bad_tensors={bad_tensors}"
        )
        if bad_first is not None:
            print(
                f"[{group_name}] first bad tensor: {bad_first}, "
                f"non_finite_count={bad_first_non_finite}"
            )


def run_multimodal_logits_compare(
    model_path: str,
    model_name: str,
    image_url: str,
) -> None:
    print("\n=== Check 3: multimodal vs text-only logits (single forward) ===")
    import requests
    from PIL import Image
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    tokenizer, model, image_processor, _max_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # A) Text-only forward (no image token)
    text_prompt = "What is shown in this image? Answer briefly."
    txt = tokenizer(text_prompt, return_tensors="pt")
    txt_ids = txt["input_ids"].to(model_device)
    txt_mask = txt["attention_mask"].to(model_device)
    with torch.inference_mode():
        txt_out = model(
            input_ids=txt_ids,
            attention_mask=txt_mask,
            return_dict=True,
        )
    _print_last_token_stats("text-only", txt_out.logits, tokenizer)

    # B) Multimodal forward (with image token + image)
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    mm_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model_device)
    mm_mask = torch.ones_like(mm_ids)

    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=model_dtype, device=model_device) for _image in image_tensor]
    image_sizes = [image.size]

    with torch.inference_mode():
        mm_out = model(
            input_ids=mm_ids,
            attention_mask=mm_mask,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * mm_ids.shape[0],
            return_dict=True,
        )
    _print_last_token_stats("multimodal", mm_out.logits, tokenizer)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hard checks for merged LLaVA model: mm_projector presence + text-only generation sanity."
    )
    ap.add_argument(
        "--model-path",
        required=True,
        help="Path to merged model directory (contains model.safetensors.index.json).",
    )
    ap.add_argument(
        "--model-name",
        default="qwen3",
        help="Model name hint for llava loader branch, e.g. qwen3.",
    )
    ap.add_argument(
        "--prompt",
        default="Please introduce yourself in one sentence.",
        help="Prompt used for text-only sanity generation.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens for text-only generation.",
    )
    ap.add_argument(
        "--skip-text",
        action="store_true",
        help="Only check weights/index, skip text generation.",
    )
    ap.add_argument(
        "--skip-mm-compare",
        action="store_true",
        help="Skip multimodal-vs-text logits comparison.",
    )
    ap.add_argument(
        "--image-url",
        default="https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true",
        help="Image URL for multimodal logits check.",
    )
    ap.add_argument(
        "--check-params-finite",
        action="store_true",
        help="Check whether multimodal parameter tensors contain NaN/Inf.",
    )
    args = ap.parse_args()

    ok_mm = check_mm_projector_keys(args.model_path)
    if not ok_mm:
        print("\n[RESULT] mm_projector missing in merged weights.")
    else:
        print("\n[RESULT] mm_projector exists in merged weights.")

    if not args.skip_text:
        try:
            run_text_only_sanity(
                model_path=args.model_path,
                model_name=args.model_name,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception:
            print("\n[ERROR] text-only sanity generation failed with exception:")
            traceback.print_exc()
            return 2

    if not args.skip_mm_compare:
        try:
            run_multimodal_logits_compare(
                model_path=args.model_path,
                model_name=args.model_name,
                image_url=args.image_url,
            )
        except Exception:
            print("\n[ERROR] multimodal-vs-text logits compare failed with exception:")
            traceback.print_exc()
            return 3

    if args.check_params_finite:
        try:
            run_parameter_finite_check(
                model_path=args.model_path,
                model_name=args.model_name,
            )
        except Exception:
            print("\n[ERROR] parameter finite check failed with exception:")
            traceback.print_exc()
            return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
