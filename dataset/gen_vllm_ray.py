import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from datasets import load_dataset
from dataset.dataset_utils import *
import argparse
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
    parser.add_argument("--output_dir_path", type=str, default="./outputs/",
                       help="Output directory for processed results")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                       help="Save checkpoint every N samples (default: 100)")
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
    parser.add_argument("--top_k", type=int, default=20,
                       help="Top-k sampling parameter")
    # vLLM Parallel parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Number of GPUs to use for pipeline parallelism")
    
    # vLLM Engine parameters
    parser.add_argument("--quantization", type=str, default=None,
                       help="Quantization method (e.g., awq, gptq, fp8, None)")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Data type for model weights (e.g., float16, bfloat16, auto)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization ratio (default: 0.6)")
    parser.add_argument("--enable_chunked_prefill", action="store_true", default=True,
                       help="Enable chunked prefill")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                       help="Trust remote code in model loading")
    parser.add_argument("--enable_reasoning", action="store_true", default=False,
                       help="Enable reasoning parser (default: False)")
    parser.add_argument("--reasoning_parser", type=str, default="deepseek_r1",
                       help="Reasoning parser method")
    
    # Input data parameters
    parser.add_argument("--task", type=str, default="llavacot", choices=["llavacot"],
                       help="Type of dataset to process")
    parser.add_argument("--parquet_dir_path", type=str, required=True,
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
    os.makedirs(args.output_dir_path, exist_ok=True)
    
    return args

def init_ray(address: str = None, log_to_driver: bool = False, show_progress: bool = True, num_cpus: int = None):
    """
    Initialize Ray cluster. If address is None, start in single-node mode.
    """
    if address:
        ray.init(address=address, ignore_reinit_error=True, log_to_driver=log_to_driver, num_cpus=num_cpus)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=log_to_driver, num_cpus=num_cpus)
    
    # 强制启用进度条，即使在非交互式环境中
    ray.data.DataContext.get_current().enable_progress_bars = True
    print("✅ Ray Data progress bars forcefully enabled")


def load_model(
    model_source: str,
    preprocess_fn=None,
    postprocess_fn=None,
    concurrency: int = 1,
    batch_size: int = 8,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int = 4096,
    max_model_len: int = 4096,
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    quantization: str = None,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.6,
    enable_reasoning: bool = False,
    reasoning_parser: str = "deepseek_r1",
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
        "dtype": dtype,
    }
    
    # Only add to engine_kwargs if quantization is not None
    if quantization is not None:
        engine_kwargs["quantization"] = quantization
        
    # Add reasoning parser parameters
    if enable_reasoning:
        engine_kwargs["enable_reasoning"] = enable_reasoning
        engine_kwargs["reasoning_parser"] = reasoning_parser

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
        chat_template_kwargs={"enable_thinking": True},  # 启用思维模式
    )

    # Build processor ONCE so the heavy model loads only once.
    processor = build_llm_processor(
        config,
        preprocess=preprocess_fn,
        postprocess=postprocess_fn,
    )

    # Return lightweight handler that simply feeds datasets to the same processor.
    def handle_dataset(dataset):
        return processor(dataset)

    return handle_dataset

def preprocess(row):
    system_prompt = SYSTEM_PROMPT_LLAVACOT
    user_prompt = USER_PROMPT_LLAVACOT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": user_prompt + "\n" + row["conversations"]
        },
    ]
    return {
        "messages": messages,
        "sampling_params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    }
def postprocess(row):
    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "generated_text": row["generated_text"],
    }

def process_dataset_with_checkpoints(
    dataset, 
    vlm_handler, 
    checkpoint_interval, 
    output_dir_path,
    show_sample_output=True,
    max_sample_display=5
):
    import os
    
    total_samples = dataset.count()
    print(f"Total samples to process: {total_samples}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    
    # 创建输出目录
    os.makedirs(output_dir_path, exist_ok=True)
    
    if total_samples <= checkpoint_interval:          # 只有一批
        batches = [dataset]                           # 直接把完整数据集当作第一批
    else:                                             # 多批
        indices = list(range(checkpoint_interval, total_samples, checkpoint_interval))
        batches = dataset.split_at_indices(indices)   # 顺序切分

    processed_count = 0
    all_results = []
    
    for batch_idx, batch_ds in enumerate(batches, 1):
        print(f"\n==== Batch {batch_idx}/{len(batches)} ====")
        result_ds = vlm_handler(batch_ds)
        batch_results = list(result_ds.iter_rows())
        all_results.extend(batch_results)
        processed_count += len(batch_results)
        
        # 输出当前批次结果（可选）
        if show_sample_output:
            print(f"Batch {batch_idx} results:")
            for sample in batch_results[:max_sample_display]:
                print(f"Generated Text: {sample['generated_text']!r}")
                print("-" * 40)
        
        # 保存当前批次的checkpoint
        checkpoint_path = os.path.join(output_dir_path, f"llavacot_trp_checkpoint_batch_{batch_idx}")
        batch_result_ds = ray.data.from_items(batch_results)
        # 强制合并为单个文件
        batch_result_ds = batch_result_ds.repartition(1)
        batch_result_ds.write_parquet(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Processed: {processed_count}/{total_samples} samples")
    
    # 保存最终完整结果
    print(f"\n{'='*60}")
    print("Saving final complete results...")
    final_result_ds = ray.data.from_items(all_results)
    # 强制合并为单个文件
    final_result_ds = final_result_ds.repartition(1)
    final_output_path = os.path.join(output_dir_path, "llavacot_trp_results")
    final_result_ds.write_parquet(final_output_path)
    print(f"Final results saved to: {final_output_path}")
    print(f"Total processed samples: {len(all_results)}")
    
    return all_results

if __name__ == "__main__":
    args = parse_args()

    # 0. Set up logging configuration
    print("="*60)
    print("Setting up logging...")
    setup_logging(
        log_level=args.log_level,
    )

    try:
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
        
        # 短暂等待确保清理完成
        time.sleep(2)
        
        init_ray(address=None, log_to_driver=not args.disable_log_to_driver, show_progress=True, num_cpus=args.num_workers)
        ctx = ray.data.DataContext.get_current()

        ctx.execution_options.preserve_order = True
        # NOTE: Increase the time Ray waits for vLLM GPU actors to start. Model loading of large checkpoints can take >10 min.
        # Formula: 10 min per tensor-parallel shard.
        ctx.wait_for_min_actors_s = 60 * 10 * args.tensor_parallel_size
        
        task = args.task
        parquet_dir_path = args.parquet_dir_path

        # 2. Read and prepare Dataset
        print("="*60)
        print("Reading dataset...")
        if task == "llavacot":
            ds = ray_prepare_data_llavacot(parquet_dir_path)
        else:
            raise ValueError(f"Invalid task: {task}")

        print("Dataset schema:", ds.schema())   # {'image': binary, 'image_id': str}
        print("Dataset Size:", ds.count())

        # TODO: 限制数据集大小（如果只想处理少量数据）
        limited_ds = ds

        print("="*60)
        print("Loading model:", args.model_source)
        vlm_handler = load_model(
            model_source=args.model_source,
            preprocess_fn=preprocess,
            postprocess_fn=postprocess,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            quantization=args.quantization,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_reasoning=args.enable_reasoning,
            reasoning_parser=args.reasoning_parser,
        )

        
        # 5. Call processing function, start parallel multimodal inference
        print("="*60)
        print("Model loaded, Start to generate...")
        
        # 使用函数进行分批处理
        checkpoint_interval = args.checkpoint_interval
        output_dir_path = args.output_dir_path
        process_dataset_with_checkpoints(
            limited_ds,
            vlm_handler,
            preprocess,
            postprocess,
            checkpoint_interval,
            output_dir_path
        )

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print("Cleaning up due to error...")
    
    finally:
        # 8. 无论如何都要清理资源
        print("="*60)
        print("Shutting down Ray and cleaning up...")
        try:
            if ray.is_initialized():
                ray.shutdown()
            print("Ray shutdown completed")
        except:
            pass
        
        # 清理 CUDA 缓存
        try:
            import torch
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        except:
            pass
        
        print("Cleanup completed.")
