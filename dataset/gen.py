import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.prompts import TASK_PROMPTS
from dataset.dataset_utils import create_batch_messages

def create_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL Image Description Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Model name or path (default: %(default)s)"
    )
    parser.add_argument(
        "--use_flash_attention", 
        action="store_true",
        help="Use flash attention for better performance"
    )
    
    # Input configuration
    parser.add_argument(
        "--task", 
        type=str,
        default="image_captioning",
        help="Task to perform (default: %(default)s)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing multiple images (default: %(default)s)"
    )
    
    # Generation configuration
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=50,
        help="Maximum number of new tokens to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use for inference (default: %(default)s)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save output to file (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def load_model(model_name, use_flash_attention=False):
    """Load the Qwen2.5-VL model and processor"""
    print(f"Loading model: {model_name}")
    
    if use_flash_attention:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
    
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model and processor loaded successfully")
    return model, processor


def generate_single_response(model, processor, messages, max_new_tokens=128, device="cuda"):
    """Generate response for single image"""
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def generate_batch_responses(model, processor, batch_messages, max_new_tokens=128, device="cuda"):
    """Generate responses for batch of images"""
    # Preparation for batch inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Batch Inference
    print(f"Generating responses for {len(batch_messages)} images...")
    with torch.no_grad():  # 添加这行节省内存
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    # 不用accelerator，直接加载模型
    model, processor = load_model(args.model_name, args.use_flash_attention)
    
    # 检查模型分布情况
    if hasattr(model, 'hf_device_map'):
        print("Model device map:", model.hf_device_map)
    
    # 模型会自动分布到多张卡
    url = "/home/muzammal/Projects/TRIG/demo.jpg"
    all_image_paths = [url] * 12
    
    print(f"Total images to process: {len(all_image_paths)}")
    print(f"Batch size: {args.batch_size}")
    
    all_responses = []
    
    # 一个进程处理所有图片，按batch分批
    batch_size = args.batch_size
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_paths)} images")
        
        batch_messages = create_batch_messages(batch_paths, TASK_PROMPTS[args.task])
        responses = generate_batch_responses(
            model, 
            processor, 
            batch_messages,
            max_new_tokens=args.max_new_tokens,
            device="cuda"  # 或者根据需要调整
        )
        all_responses.extend(responses)
    
    # 显示结果
    print("\n" + "="*50)
    print("BATCH RESPONSES:")
    print("="*50)
    for i, (image_path, response) in enumerate(zip(all_image_paths, all_responses), 1):
        print(f"\nImage {i}: {image_path}")
        print("-"*30)
        print(response)
        print("-"*30)


if __name__ == "__main__":
    main()