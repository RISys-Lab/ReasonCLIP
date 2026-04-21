from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

pretrained = "/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_sft"
model_name = "qwen3"
vision_tower_name = "fesvhtr/clip-r-336-s1-run1215-1280"
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
)  # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()

# Hotfix: override vision tower weights with fresh HF weights.
# The merged checkpoint may contain stale/corrupted vision_tower params.
vt = model.get_vision_tower()
vt_model = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float32).to(vt.device)
vt.vision_tower = vt_model
vt.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
image_processor = vt.image_processor

url = "/home/localadmin/bz/ReasonCLIP/data/Urban1k/Urban1k/image/1.jpg"
image = Image.open(url).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
model_dtype = next(model.parameters()).dtype
image_tensor = [_image.to(dtype=model_dtype, device=device) for _image in image_tensor]

conv_template = "qwen_1_5" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
attention_mask = torch.ones_like(input_ids)
image_sizes = [image.size]


outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    max_new_tokens=256,
    return_dict_in_generate=True,
    output_scores=True,
    modalities=["image"] * input_ids.shape[0],
)

# Robustly extract generated tokens for multimodal inputs_embeds path.
# Number of generated steps is len(scores), which avoids prompt-length mismatch.
sequences = outputs.sequences
gen_len = len(outputs.scores) if outputs.scores is not None else max(sequences.shape[1] - input_ids.shape[1], 0)
gen_ids = sequences[:, -gen_len:] if gen_len > 0 else sequences[:, 0:0]
text_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(text_outputs[0].strip())