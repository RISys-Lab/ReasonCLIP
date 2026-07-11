#!/bin/bash
#SBATCH --job-name=qwen3_8b_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=qwen3_8b_smoke_%j.out
#SBATCH --error=qwen3_8b_smoke_%j.err
#SBATCH --account=kuin0164
#SBATCH --mem=96G

set -euo pipefail

REPO_ROOT="/dpc/kuin0164/zsc/ReasonCLIP"
VENV="/dpc/kuin0164/zsc/venv/llm"

export HF_HOME="${HF_HOME:-/dpc/kuin0164/zsc/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dpc/kuin0164/zsc/venv/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/dpc/kuin0164/zsc/venv/uv-python}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load profile/deeplrn 2>/dev/null || true
module load cuda/13.0 2>/dev/null || true

source "${VENV}/bin/activate"
cd "${REPO_ROOT}"

python -u - <<'PY'
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
n_runs = int(os.environ.get("N_RUNS", "10"))
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "96"))
prompt = os.environ.get(
    "PROMPT",
    "In one concise paragraph, explain why contrastive vision-language models are useful.",
)

job_id = os.environ.get("SLURM_JOB_ID", "local")
out_dir = Path("outputs/qwen3_8b_smoke")
out_dir.mkdir(parents=True, exist_ok=True)
jsonl_path = out_dir / f"{job_id}.jsonl"

print(f"model_id={model_id}")
print(f"hf_home={os.environ.get('HF_HOME')}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
print(f"jsonl_path={jsonl_path}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

def render_prompt(user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


text = render_prompt(prompt)
device = next(param.device for param in model.parameters() if param.device.type != "meta")

with jsonl_path.open("w", encoding="utf-8") as f:
    for run_idx in range(1, n_runs + 1):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[:, inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        record = {"run": run_idx, "prompt": prompt, "response": response}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

        print(f"\n===== RUN {run_idx}/{n_runs} =====")
        print(response)

print(f"\nWrote {jsonl_path}")
PY
