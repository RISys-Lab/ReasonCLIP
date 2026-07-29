import sys
import os
import argparse
import torch
from transformers import AutoProcessor
from llavaonevision1_5.modeling_llavaonevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)


def apply_training_chat_template(messages):
    """Format multimodal messages exactly as the SFT dataset does."""
    system_content = SYSTEM_MESSAGE
    turns = messages
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        turns = messages[1:]

    formatted = (
        f"{DEFAULT_IM_START_TOKEN}system\n"
        f"{system_content}{DEFAULT_IM_END_TOKEN}\n"
    )
    for message in turns:
        content = message["content"]
        if isinstance(content, str):
            formatted_content = content
        else:
            parts = []
            for item in content:
                if item["type"] == "image":
                    parts.append(
                        f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}"
                    )
                elif item["type"] == "video":
                    parts.append(
                        f"{VISION_START_TOKEN}{DEFAULT_VIDEO_TOKEN}{VISION_END_TOKEN}"
                    )
                elif item["type"] == "text":
                    parts.append(item["text"])
            formatted_content = "".join(parts)

        formatted += (
            f"{DEFAULT_IM_START_TOKEN}{message['role']}\n"
            f"{formatted_content}{DEFAULT_IM_END_TOKEN}\n"
        )

    return f"{formatted}{DEFAULT_IM_START_TOKEN}assistant\n"


@torch.inference_mode()
def generate_for_messages(model, processor, messages, max_new_tokens):
    """
    A helper function to run the full generation pipeline for a given set of messages.
    """
    # --- Preparation for inference ---
    # The saved Qwen3 text template drops structured multimodal content.
    text = apply_training_chat_template(messages)
    
    # Process visual information (images/videos) from the messages
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Combine text, image, and video inputs into a single model input
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)
    image_token_count = (
        inputs.input_ids == model.config.image_token_id
    ).sum().item()
    print(f"> Expanded image tokens: {image_token_count}")

    # --- Inference: Generation of the output ---
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=151645,
        do_sample=False,
    )
    
    # Trim the generated IDs to remove the prompt portion
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode the output text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Print the prompt and the generated text
    prompt = messages[0]['content'][1]['text']
    print(f"\n> Prompt: {prompt}")
    print(f"> Generated Text: {output_text[0].strip()}")


def main(args):
    """
    Main function to load the model and generate captions for an image in English and Chinese.
    """
    print(f"Loading model from path: {args.model_path}")
    
    # Load the model and processor from the specified path
    # device_map="auto" will handle placing the model on available GPUs
    model = LLaVAOneVision1_5_ForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto",
        trust_remote_code=False # Recommended for custom models
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=False
    )
    print("✓ Model and processor loaded successfully.")
    model.eval()

    for image_index, image_path in enumerate(args.image_paths, start=1):
        print(f"\n=== Image {image_index}/{len(args.image_paths)}: {image_path} ===")

        # --- Test with English Prompt ---
        print("\n--- Testing with English Prompt ---")
        english_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]
        generate_for_messages(
            model, processor, english_messages, args.max_new_tokens
        )

        # --- Test with Chinese Prompt ---
        print("\n--- Testing with Chinese Prompt ---")
        chinese_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "请用中文详细描述这张图片。"},
                ],
            }
        ]
        generate_for_messages(
            model, processor, chinese_messages, args.max_new_tokens
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from an image using a Qwen2-VL model.")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the pretrained model and processor."
    )
    parser.add_argument(
        "--image-path",
        dest="image_paths",
        type=str,
        nargs="+",
        default=["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
        help="One or more local image paths or URLs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens generated for each prompt.",
    )
    
    args = parser.parse_args()
    main(args)
    
