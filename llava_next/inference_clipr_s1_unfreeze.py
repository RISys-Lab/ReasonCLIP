from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from PIL import Image
import copy
import torch


pretrained = "/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_s1_unfreeze_sft"
model_name = "qwen3"
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

# This checkpoint already contains the S1-unfrozen CLIP-R vision tower.
# Do not overwrite it with fresh remote CLIP-R weights.
vt = model.get_vision_tower()
if image_processor is None:
    image_processor = vt.image_processor

image_path = "/home/localadmin/bz/ReasonCLIP/data/Urban1k/Urban1k/image/1.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
model_dtype = next(model.parameters()).dtype
image_tensor = [_image.to(dtype=model_dtype, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"
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

sequences = outputs.sequences
gen_len = len(outputs.scores) if outputs.scores is not None else max(sequences.shape[1] - input_ids.shape[1], 0)
gen_ids = sequences[:, -gen_len:] if gen_len > 0 else sequences[:, 0:0]
text_outputs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print(text_outputs[0].strip())
