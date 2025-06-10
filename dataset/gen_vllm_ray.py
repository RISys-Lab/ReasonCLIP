import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from datasets import load_dataset
from dataset.dataset_utils import ray_prepare_data_llavacot, SYSTEM_PROMPT_LLAVACOT
import argparse
import os
from PIL import Image
import logging
import time

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration based on arguments"""
    
    # Convert string to logging level
    level = getattr(logging, log_level.upper())
    
    # Set basic logging configuration
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure specific loggers (enabled by default)
    vllm_logger = logging.getLogger('vllm')
    vllm_logger.setLevel(level)
    print(f"vLLM detailed logging enabled at {log_level} level")

    ray_logger = logging.getLogger('ray.data')
    ray_logger.setLevel(level)
    print(f"Ray Data detailed logging enabled at {log_level} level")


assert Version(ray.__version__) >= Version("2.44.1"), "Need Ray 2.44.1 or higher for vLLM support"

def parse_args():
    """Parse command line arguments for VLM processing"""
    
    parser = argparse.ArgumentParser(description="Generate with multimodal data by vllm")
    
    # Model parameters
    parser.add_argument("--model_source", type=str, default="Qwen/Qwen3-32B", 
                       help="Model name or path")
    parser.add_argument("--output_path", type=str, default="./outputs/",
                       help="Output directory for processed results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model sequence length")
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096,
                       help="Maximum number of batched tokens")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    
    # vLLM Parallel parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Number of GPUs to use for pipeline parallelism")
    
    # vLLM Engine parameters
    parser.add_argument("--quantization", type=str, default=None,
                       help="Quantization method (e.g., awq, gptq, fp8, None)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio (default: 0.6)")
    parser.add_argument("--enable_chunked_prefill", action="store_true", default=True,
                       help="Enable chunked prefill")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                       help="Trust remote code in model loading")
    
    # Input data parameters
    parser.add_argument("--task", type=str, default="llavacot", choices=["llavacot"],
                       help="Type of dataset to process")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input data (file pattern or directory)")
    parser.add_argument("--concurrency", type=int, default=2,
                       help="Number of vLLM instances to run in parallel")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2,
                       help="Number of workers for data loading")
    
    # System parameters
    parser.add_argument("--ray_address", type=str, default=None,
                       help="Ray cluster address (None for local)")
    
    # Logging parameters
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Global logging level (default: %(default)s)")
    parser.add_argument("--disable_vllm_logs", action="store_true",
                       help="Disable detailed vLLM logging (default: enabled)")
    parser.add_argument("--disable_ray_logs", action="store_true", 
                       help="Disable detailed Ray Data logging (default: enabled)")
    parser.add_argument("--disable_log_to_driver", action="store_true",
                       help="Disable Ray log output to driver (default: enabled)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    return args

def init_ray(address: str = None, log_to_driver: bool = False, show_progress: bool = True, num_cpus: int = None):
    """
    Initialize Ray cluster. If address is None, start in single-node mode.
    """
    if address:
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=log_to_driver, num_cpus=num_cpus)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=log_to_driver, num_cpus=num_cpus)
    if not show_progress:
        ray.data.DataContext.get_current().enable_progress_bars = False


def load_model(
    model_source: str,
    concurrency: int = 1,
    batch_size: int = 8,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 4096,
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    quantization: str = None,
    gpu_memory_utilization: float = 0.6,
):

    # Build engine_kwargs, only include non-None parameters
    engine_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "enable_chunked_prefill": enable_chunked_prefill,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    
    # Only add to engine_kwargs if quantization is not None
    if quantization is not None:
        engine_kwargs["quantization"] = quantization

    config = vLLMEngineProcessorConfig(
        model_source=model_source,
        # The kwargs to pass to the vLLM engine. https://docs.vllm.ai/en/v0.7.3/serving/engine_args.html
        engine_kwargs=engine_kwargs,
        runtime_env={
            "env_vars": {
                "VLLM_USE_V1": "1",  # Use vLLM v1 engine
            }
        },
        concurrency=concurrency,
        batch_size=batch_size,
        accelerator_type=None,
        apply_chat_template=True,
        has_image=True,
    )

    def handle_dataset(dataset, preprocess_fn, postprocess_fn):
        processor = build_llm_processor(
            config,
            preprocess=preprocess_fn,
            postprocess=postprocess_fn,
        )
        return processor(dataset)

    return handle_dataset

def preprocess(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_LLAVACOT},
        {
            "role": "user", 
            "content": row["conversations"]
        },
    ]
    return {
        "messages": messages,
        "sampling_params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        },
    }
def postprocess(row):
    return {
        "id": row["id"],
        "image": row["image"],
        "generated_text": row["generated_text"],
    }




if __name__ == "__main__":
    args = parse_args()

    # 0. Set up logging configuration
    print("="*60)
    print("Setting up logging...")
    setup_logging(
        log_level=args.log_level,
    )

    # 1. Force clean up and initialize Ray (single-node mode)
    print("="*60)
    print("Cleaning up and initializing Ray...")
    
    # Force close any existing Ray instances
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except:
        pass
    
    time.sleep(2)
    
    init_ray(address=None, log_to_driver=not args.disable_log_to_driver, show_progress=True, num_cpus=args.num_workers)
    task = args.task
    data_path = args.data_path

    # 2. Read and prepare Dataset
    print("="*60)
    print("Reading dataset...")
    if task == "llavacot":
        ds = ray_prepare_data_llavacot(data_path)
    else:
        raise ValueError(f"Invalid task: {task}")

    print("Dataset schema:", ds.schema())   # {'image': binary, 'image_id': str}
    print("Dataset Size:", ds.count())

    print("="*60)
    print("Loading model:", args.model_source)
    vlm_handler = load_model(
        model_source=args.model_source,  
        concurrency=args.concurrency, 
        batch_size=args.batch_size,                 
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    
    # 5. Call processing function, start parallel multimodal inference
    print("="*60)
    print("Model loaded, Start to generate...")
    result_ds = vlm_handler(ds, preprocess, postprocess)

    # 6. Output the first 5 rows to the console (local debug)
    print("="*60)
    print("Outputting results...")
    for sample in result_ds.take(20):
        print(f"Answer: {sample['answer']!r}")
        print("-" * 60)

    # 7. Write complete results back to S3 (Parquet format)
    result_ds.write_parquet(args.output_path)

    # 8. Close Ray
    ray.shutdown()
