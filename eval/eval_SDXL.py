import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, AutoTokenizer

device = "cuda"

# --------------------------------------------------
# 1️⃣  加载 SDXL Base
# --------------------------------------------------
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
base.to(device)


# --------------------------------------------------
# 2️⃣  加载你自己的 CLIP
# --------------------------------------------------
your_clip_path = "fesvhtr/clip-r-336-s2-run0204-505"   # ← 改成你的路径

text_encoder = CLIPTextModel.from_pretrained(
    your_clip_path,
    torch_dtype=torch.float16
).to(device)

tokenizer = AutoTokenizer.from_pretrained(your_clip_path)


# --------------------------------------------------
# 3️⃣  替换 SDXL 的 text_encoder + tokenizer
# --------------------------------------------------
base.text_encoder = text_encoder
base.tokenizer = tokenizer


# --------------------------------------------------
# 4️⃣  加载 Refiner（共享 text_encoder_2 & VAE）
# --------------------------------------------------
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,  # 不改
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to(device)


# --------------------------------------------------
# 5️⃣  推理参数
# --------------------------------------------------
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"


# --------------------------------------------------
# 6️⃣  运行 Base
# --------------------------------------------------
latents = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images


# --------------------------------------------------
# 7️⃣  运行 Refiner
# --------------------------------------------------
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=latents,
).images[0]


# --------------------------------------------------
# 8️⃣  保存
# --------------------------------------------------
image.save("result.png")