import argparse
import copy
import os
import sys

import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import CLIPImageProcessor, CLIPVisionModel

# 让脚本可直接运行：自动把 llava_next 加入 PYTHONPATH / sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAVA_NEXT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if LLAVA_NEXT_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_NEXT_ROOT)
os.environ["PYTHONPATH"] = LLAVA_NEXT_ROOT + ":" + os.environ.get("PYTHONPATH", "")

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def get_gt_answer(row: dict) -> str:
    """VQAv2: row["answers"][0]["answer"]"""
    return str(row["answers"][0]["answer"]).strip()


def is_match(pred: str, gt: str) -> bool:
    """精确匹配"""
    return pred.strip().lower() == gt.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model-preset", type=str, default="clipr", choices=["clipr", "clip"])
    args = parser.parse_args()

    dataset_name = "lmms-lab/VQAv2"
    config = get_dataset_config_names(dataset_name)[0]
    ds = load_dataset(dataset_name, config)
    split_name = "train"
    split_ds = ds[split_name]
    total = len(split_ds)

    model_name = "qwen3"
    if args.model_preset == "clip":
        # 对齐 inference_clip.py
        pretrained = "/home/localadmin/bz/CLIP-R/llava_next/checkpoints/merged/clip_qwen3_sft"
        vision_tower_name = "openai/clip-vit-large-patch14-336"
    else:
        # 对齐 inference_clipr.py
        pretrained = "/home/localadmin/bz/CLIP-R/llava_next/checkpoints/merged/clipr_qwen3_sft"
        vision_tower_name = "fesvhtr/clip-r-336-s1-run1215-1280"
    device = "cuda"

    tokenizer, model, image_processor, _max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        device_map="auto",
        multimodal=True,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
    )
    model.eval()
    model.tie_weights()

    # 和 inference_clipr.py 对齐：覆盖 vision tower，避免 merged 中坏权重。
    vt = model.get_vision_tower()
    vt_model = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float32).to(vt.device)
    vt.vision_tower = vt_model
    vt.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    image_processor = vt.image_processor

    print(f"dataset={dataset_name}")
    print(f"config={config}")
    print(f"split={split_name}")
    print(f"start_sample_idx={args.sample_idx}")
    print(f"num_samples={args.num_samples}")
    print(f"model_preset={args.model_preset}")
    print(f"pretrained={pretrained}")
    print(f"vision_tower={vision_tower_name}")
    print("-" * 80)

    hit = 0
    valid = 0
    end_idx = min(args.sample_idx + args.num_samples, total)
    model_dtype = next(model.parameters()).dtype

    for i in range(args.sample_idx, end_idx):
        row = split_ds[i]
        image = row["image"].convert("RGB")
        question = row["question"]
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_t.to(dtype=model_dtype, device=device) for _t in image_tensor]

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        prompt_q = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], prompt_q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                modalities=["image"] * input_ids.shape[0],
            )

        sequences = outputs.sequences
        gen_len = len(outputs.scores)
        gen_ids = sequences[:, -gen_len:]
        pred = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        gt = get_gt_answer(row)
        match = is_match(pred, gt)
        valid += 1
        if match:
            hit += 1

        print(f"[{i}] question_id={row.get('question_id', '')} category={row.get('category', '')}")
        print(f"  question={question}")
        print(f"  gt={gt}")
        print(f"  pred={pred}")
        print(f"  match={match}")
        print("-" * 80)

    acc = hit / valid
    print(f"summary: hit={hit}, valid={valid}, acc={acc:.4f}")


if __name__ == "__main__":
    main()
