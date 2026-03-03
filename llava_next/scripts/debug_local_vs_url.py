#!/usr/bin/env python3
"""调试：本地路径 vs URL 加载图片，为何本地会空输出"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
import requests
from io import BytesIO
import torch

# 同一张图：先通过 URL 下载，再分别用两种方式加载
RADAR_URL = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
LOCAL_COCO = "/home/localadmin/bz/CLIP-R/data/llava-sft-data/images/coco/train2017/000000000009.jpg"
TEMP_RADAR = "/tmp/llava_radar_debug.jpg"


def load_via_url(url):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    img = Image.open(r.raw).convert("RGB")
    return img


def load_via_path(path):
    img = Image.open(path).convert("RGB")
    return img


def main():
    print("=== 1. 检查文件存在 ===")
    print(f"COCO 本地存在: {os.path.exists(LOCAL_COCO)}")
    if not os.path.exists(LOCAL_COCO):
        print("COCO 路径不存在，请检查")
        return

    print("\n=== 2. 加载图片并对比 ===")
    # 下载 radar 到本地，用于公平对比
    img_url = load_via_url(RADAR_URL)
    img_url.save(TEMP_RADAR)
    img_local_radar = load_via_path(TEMP_RADAR)
    img_local_coco = load_via_path(LOCAL_COCO)

    print(f"Radar (URL):     size={img_url.size}, mode={img_url.mode}")
    print(f"Radar (本地):    size={img_local_radar.size}, mode={img_local_radar.mode}")
    print(f"COCO (本地):     size={img_local_coco.size}, mode={img_local_coco.mode}")

    # 检查 Radar URL vs 本地是否一致
    import numpy as np
    arr_url = np.array(img_url)
    arr_local = np.array(img_local_radar)
    print(f"Radar URL vs 本地 像素一致: {np.array_equal(arr_url, arr_local)}")

    print("\n=== 3. 加载模型并分别推理 ===")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from transformers import CLIPVisionModel, CLIPImageProcessor
    import copy

    pretrained = "/home/localadmin/bz/CLIP-R/llava_next/checkpoints/merged/clipr_qwen3_sft"
    vision_tower_name = "fesvhtr/clip-r-336-s1-run1215-1280"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, "qwen3",
        device_map="auto", multimodal=True,
        torch_dtype="bfloat16", attn_implementation="sdpa",
    )
    vt = model.get_vision_tower()
    vt.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float32).to(vt.device)
    vt.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    image_processor = vt.image_processor
    model.eval()

    def run_inference(image, name):
        image_tensor = process_images([image], image_processor, model.config)
        model_dtype = next(model.parameters()).dtype
        image_tensor = [_t.to(dtype=model_dtype, device=device) for _t in image_tensor]
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            cont = model.generate(
                input_ids, attention_mask=attention_mask,
                images=image_tensor, image_sizes=[image.size],
                do_sample=False, max_new_tokens=128,
                repetition_penalty=1.2, modalities=["image"] * input_ids.shape[0],
            )
        gen_ids = cont[:, input_ids.shape[1]:]
        text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        print(f"\n[{name}]")
        print(f"  gen_ids 长度: {gen_ids.shape[1]}")
        print(f"  gen_ids 前10: {gen_ids[0, :10].tolist()}")
        print(f"  输出: '{text}'")
        return text

    run_inference(img_url, "Radar (URL 加载)")
    run_inference(img_local_radar, "Radar (本地加载，同一张图)")
    run_inference(img_local_coco, "COCO (本地加载)")

    print("\n=== 结论 ===")
    print("若 Radar 本地也空 → 问题在本地加载方式")
    print("若 Radar 本地正常、COCO 空 → 问题在 COCO 图片内容或模型对该类图敏感")


if __name__ == "__main__":
    main()
