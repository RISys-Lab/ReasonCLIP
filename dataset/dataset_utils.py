import os
import ray
from datasets import load_dataset
import gc
import datetime
import re


def create_batch_messages(image_paths, text_prompt):
    """Create batch messages for multiple images with same prompt"""
    messages = []
    for image_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        messages.append(message)
    return messages

def process_dataset_with_checkpoints(
    dataset, 
    processor, 
    task,
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
        print("="*60)
        print(f"==== Batch {batch_idx}/{len(batches)} ====")
        result_ds = processor(batch_ds)
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(output_dir_path, f"{task}_{timestamp}_checkpoint_batch_{batch_idx}")
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
    final_output_path = os.path.join(output_dir_path, f"{task}_{timestamp}_results")
    final_result_ds.write_parquet(final_output_path)
    print(f"Final results saved to: {final_output_path}")
    print(f"Total processed samples: {len(all_results)}")
    
    return all_results

def process_dataset_with_checkpoints_optimized(
    dataset, processor, task,
    checkpoint_interval, output_dir_path,
    show_sample_output=True, max_sample_display=5,
    ray_batch_size=None,
):
    import datetime, gc, os
    os.makedirs(output_dir_path, exist_ok=True)

    total = dataset.count()
    step  = ray_batch_size or min(checkpoint_interval, 8192)  # 给个上限，防爆内存
    next_ckpt = checkpoint_interval
    processed = 0
    buffer = []
    batch_idx = 0

    for batch in dataset.iter_batches(batch_format="pandas",
                                      batch_size=step,
                                      drop_empty_batches=True):
        batch_idx += 1
        out_rows = list(processor(ray.data.from_pandas(batch)).iter_rows())
        buffer.extend(out_rows)
        processed += len(out_rows)

        if show_sample_output and out_rows:
            print(f"\n⏺ Batch {batch_idx} sample:")
            for r in out_rows[:max_sample_display]:
                print("  »", r.get("generated_text", "")[:80].replace("\n"," "))
            print("-" * 40)

        # flush 条件
        if (processed >= next_ckpt and buffer) or processed == total:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir_path,
                                f"{task}_ckpt_{processed:07d}_{ts}")
            ray.data.from_items(buffer).repartition(1).write_parquet(path)
            print(f"💾 saved {path}")
            buffer.clear()
            next_ckpt += checkpoint_interval

        del batch, out_rows
        gc.collect()

    print(f"✅ done: {processed}/{total}")

