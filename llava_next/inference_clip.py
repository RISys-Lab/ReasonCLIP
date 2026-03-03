"""CLIP 版推理脚本，测试合并后的 clip_qwen3_sft"""
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image
import requests
import copy
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

pretrained = "/home/localadmin/bz/CLIP-R/llava_next/checkpoints/merged/clip_qwen3_sft"
model_name = "qwen3"
vision_tower_name = "openai/clip-vit-large-patch14-336"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained,
    None,
    model_name,
    device_map=device_map,
    multimodal=True,
    torch_dtype="bfloat16",
    attn_implementation="sdpa",
)

model.eval()
model.tie_weights()

# 强制用 HF 上干净的 vision tower，避免 merged 里潜在坏权重
vt = model.get_vision_tower()
vt_model = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float32).to(vt.device)
vt.vision_tower = vt_model
vt.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
image_processor = vt.image_processor

# 支持本地路径和 URL
image_path = "/home/localadmin/bz/CLIP-R/data/llava-sft-data/images/ai2d/abc_images/5.png"
if image_path.startswith("http"):
    image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
else:
    image = Image.open(image_path).convert("RGB")

image_tensor = process_images([image], image_processor, model.config)
model_dtype = next(model.parameters()).dtype
image_tensor = [_t.to(dtype=model_dtype, device=device) for _t in image_tensor]

conv_template = "qwen_1_5"
question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
attention_mask = torch.ones_like(input_ids)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    attention_mask=attention_mask,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    max_new_tokens=256,
    repetition_penalty=1.2,
    modalities=["image"] * input_ids.shape[0],
)
gen_ids = cont[:, input_ids.shape[1]:]
text_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(text_outputs[0].strip())
