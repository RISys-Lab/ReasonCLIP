import copy
import os
import sys
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Ensure local llava_next is importable in lmms-eval runtime.
LLAVA_NEXT_ROOT = os.environ.get("LLAVA_NEXT_ROOT", "/home/localadmin/bz/ReasonCLIP/llava_next")
if LLAVA_NEXT_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_NEXT_ROOT)

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug(f"LLaVA import failed: {e}")


@register_model("llava_clipr")
class LlavaClipR(lmms):
    """
    LLaVA wrapper for CLIP-R/Qwen3 merged checkpoint.
    Keep behavior aligned with llava_next/inference_clipr.py.
    """

    def __init__(
        self,
        pretrained: str = "/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_sft",
        model_name: str = "qwen3",
        vision_tower_name: str = "fesvhtr/clip-r-336-s1-run1215-1280",
        conv_template: str = "qwen_1_5",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        tie_weights: bool = True,
        use_cache: bool = True,
        reload_vision_tower: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = "auto"
        else:
            self._device = torch.device(device)
            self.device_map = device

        self._tokenizer, self._model, image_processor, self._max_length = load_pretrained_model(
            pretrained,
            None,
            model_name,
            device_map=self.device_map,
            multimodal=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        self._model.eval()
        if tie_weights:
            self._model.tie_weights()

        vt = self._model.get_vision_tower()
        if reload_vision_tower:
            # Keep same legacy hotfix as inference_clipr.py: overwrite vision tower.
            vt_model = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float32).to(vt.device)
            vt.vision_tower = vt_model
            vt.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
            self._image_processor = vt.image_processor
        else:
            # Unfreezed checkpoints already contain the trained vision tower.
            eval_logger.info("Using vision tower weights from checkpoint; not reloading CLIPVisionModel.")
            self._image_processor = getattr(vt, "image_processor", None) or image_processor
            if self._image_processor is None:
                self._image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
                vt.image_processor = self._image_processor

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.model_dtype = next(self._model.parameters()).dtype
        self._rank = accelerator.local_process_index if accelerator.num_processes > 1 else 0
        self._world_size = accelerator.num_processes

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _flatten(self, visuals):
        if visuals is None:
            return []
        if isinstance(visuals, list):
            out = []
            for v in visuals:
                if isinstance(v, list):
                    out.extend(v)
                elif v is not None:
                    out.append(v)
            return out
        return [visuals]

    def _build_image_tensor(self, visuals):
        image_tensor = process_images(visuals, self._image_processor, self._config) if visuals else None
        if isinstance(image_tensor, list):
            image_tensor = [_img.to(dtype=self.model_dtype, device=self.device) for _img in image_tensor]
        elif image_tensor is not None:
            image_tensor = image_tensor.to(dtype=self.model_dtype, device=self.device)
        return image_tensor

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res: List[Tuple[float, bool]] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Scoring")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = self._flatten(doc_to_visual(doc))

            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(doc)
            continuation = str(continuation)

            context = contexts[0] if isinstance(contexts, list) else contexts
            if visuals and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
                context = image_tokens + "\n" + context

            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            context_prompt = conv.get_prompt()
            context_ids = tokenizer_image_token(
                context_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)

            # Append continuation as assistant answer to score log p(answer | context, image).
            conv.messages[1][1] = continuation
            full_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels[:, : context_ids.shape[1]] = -100

            image_tensor = self._build_image_tensor(visuals)
            image_sizes = [img.size for img in visuals] if visuals else None

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    modalities=["image"] * input_ids.shape[0],
                    use_cache=True,
                )

            loss = float(outputs.loss.item())

            # For multimodal inputs, exact greedy-check alignment is non-trivial due to
            # image-token expansion in prepare_inputs_labels_for_multimodal.
            is_greedy = False
            res.append((loss, is_greedy))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = self._flatten(doc_to_visual(doc))

            context = contexts[0] if isinstance(contexts, list) else contexts
            if visuals and DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * len(visuals))
                context = image_tokens + "\n" + context

            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            image_tensor = self._build_image_tensor(visuals)

            max_new_tokens = int(gen_kwargs.get("max_new_tokens", 256))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_p = gen_kwargs.get("top_p", None)
            num_beams = int(gen_kwargs.get("num_beams", 1))
            until = gen_kwargs.get("until", None)
            if isinstance(until, str):
                until = [until]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=[img.size for img in visuals] if visuals else None,
                    do_sample=temperature > 0,
                    temperature=(temperature if temperature > 0 else None),
                    top_p=(top_p if temperature > 0 else None),
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    modalities=["image"] * input_ids.shape[0],
                    use_cache=self.use_cache,
                )

            sequences = outputs.sequences
            gen_len = len(outputs.scores)
            gen_ids = sequences[:, -gen_len:] if gen_len > 0 else sequences[:, 0:0]
            answer = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            if until:
                for term in until:
                    if term:
                        answer = answer.split(term)[0]

            res.append(answer)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("llava_clipr does not support multi-round generation yet.")


@register_model("llava_clipr_unfreezed")
class LlavaClipRUnfreezed(LlavaClipR):
    """
    LLaVA CLIP-R/Qwen3 wrapper for the S1-unfreezed merged checkpoint.
    This intentionally keeps the vision tower loaded from the checkpoint instead
    of reloading a remote CLIP-R encoder.
    """

    def __init__(
        self,
        pretrained: str = "/home/localadmin/bz/ReasonCLIP/llava_next/checkpoints/merged/clipr_qwen3_s1_unfreeze_sft",
        model_name: str = "qwen3",
        vision_tower_name: str = "fesvhtr/clip-r-336-s1-run1215-1280",
        conv_template: str = "qwen_1_5",
        device: str = "cuda:0",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        tie_weights: bool = True,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            model_name=model_name,
            vision_tower_name=vision_tower_name,
            conv_template=conv_template,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            tie_weights=tie_weights,
            use_cache=use_cache,
            reload_vision_tower=False,
            **kwargs,
        )
