from transformers import AutoConfig, SiglipModel
from safetensors.torch import load_file

cfg = AutoConfig.from_pretrained("fesvhtr/siglip-r-s1-v1-1926")
m = SiglipModel(cfg)
state = load_file("/home/muzammal/.cache/huggingface/hub/models--fesvhtr--siglip-r-s1-v1-1926/snapshots/fba50d08af3913c3028e0c3b771bd5b171e490d1/model.safetensors")
missing, unexpected = m.load_state_dict(state, strict=False)
print("missing:", len(missing), missing[:20])
print("unexpected:", len(unexpected), unexpected[:20])
# 若 missing 里有 text_projection/visual_projection/vision_model/text_model 等关键项，问题即成立
keys = list(state.keys())[:30]
print(keys)
# 若看到以 "module." 或 "backbone." 开头，大概率就是命名不匹配
