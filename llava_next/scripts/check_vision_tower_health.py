#!/usr/bin/env python3
import argparse
import traceback

import requests
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


def check_parameter_finite(model: CLIPVisionModel) -> None:
    total_tensors = 0
    bad_tensors = 0
    first_bad_name = None
    first_bad_count = 0

    for name, param in model.named_parameters():
        total_tensors += 1
        finite_mask = torch.isfinite(param.detach())
        non_finite_count = int((~finite_mask).sum().item())
        if non_finite_count > 0:
            bad_tensors += 1
            if first_bad_name is None:
                first_bad_name = name
                first_bad_count = non_finite_count

    print("=== Parameter Finite Check ===")
    print(f"total tensors: {total_tensors}")
    print(f"bad tensors: {bad_tensors}")
    if first_bad_name is not None:
        print(f"first bad tensor: {first_bad_name}")
        print(f"first bad non-finite count: {first_bad_count}")


@torch.inference_mode()
def check_forward_finite(
    model: CLIPVisionModel,
    processor: CLIPImageProcessor,
    image_url: str,
    device: torch.device,
) -> None:
    image = Image.open(requests.get(image_url, stream=True, timeout=20).raw).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)

    outputs = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    last_hidden = outputs.last_hidden_state
    pooled = outputs.pooler_output

    hidden_finite_ratio = float(torch.isfinite(last_hidden).float().mean().item())
    pooled_finite_ratio = float(torch.isfinite(pooled).float().mean().item())

    print("\n=== Forward Finite Check ===")
    print(f"last_hidden_state finite ratio: {hidden_finite_ratio:.6f}")
    print(f"pooler_output finite ratio: {pooled_finite_ratio:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check CLIP vision tower parameter/forward health.")
    ap.add_argument(
        "--vision-model",
        default="fesvhtr/clip-r-336-des-run0201-949",
        help="HF vision model id/path, e.g. fesvhtr/clip-r-336-des-run0201-949",
    )
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model load dtype for health check.",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for forward check.",
    )
    ap.add_argument(
        "--image-url",
        default="https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true",
        help="Image URL for forward finite test.",
    )
    ap.add_argument(
        "--fallback-processor",
        default="openai/clip-vit-large-patch14-336",
        help="Fallback image processor id/path when target model has no preprocessor config.",
    )
    args = ap.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=== Load Vision Tower ===")
    print(f"vision model: {args.vision_model}")
    print(f"dtype: {args.dtype}")
    print(f"device: {device}")

    try:
        try:
            processor = CLIPImageProcessor.from_pretrained(args.vision_model)
        except Exception:
            print(
                f"[WARN] Failed to load image processor from {args.vision_model}, "
                f"fallback to {args.fallback_processor}"
            )
            processor = CLIPImageProcessor.from_pretrained(args.fallback_processor)
        model = CLIPVisionModel.from_pretrained(args.vision_model, torch_dtype=torch_dtype).to(device)
        model.eval()
    except Exception:
        print("[ERROR] failed to load vision tower")
        traceback.print_exc()
        return 2

    check_parameter_finite(model)
    check_forward_finite(model, processor, args.image_url, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
