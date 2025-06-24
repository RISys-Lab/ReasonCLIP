import os
import ray
from datasets import load_dataset
import gc
import datetime
import re
import glob


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


def get_last_processed_index(output_dir_path, task):
    """
    获取上次处理到的位置（已处理的样本总数）
    通过检查checkpoint文件名中的数字来确定
    """
    if not os.path.exists(output_dir_path):
        print(f"Output directory {output_dir_path} does not exist, starting from beginning")
        return 0
    
    # 查找所有checkpoint文件
    checkpoint_pattern = os.path.join(output_dir_path, f"{task}_ckpt_*.parquet")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {output_dir_path}, starting from beginning")
        return 0
    
    # 从文件名中提取已处理的样本数量
    max_processed = 0
    latest_checkpoint = None
    
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        # 匹配格式: {task}_ckpt_{processed:07d}_{timestamp}.parquet
        match = re.search(rf'{re.escape(task)}_ckpt_(\d+)_', filename)
        if match:
            processed_count = int(match.group(1))
            if processed_count > max_processed:
                max_processed = processed_count
                latest_checkpoint = file_path
    
    if max_processed > 0:
        print(f"Found latest checkpoint: {latest_checkpoint}")
        print(f"Last processed index: {max_processed}")
        return max_processed
    else:
        print("No valid checkpoint files found, starting from beginning")
        return 0


def get_dataset_slice_from_index(dataset, start_index, total_count=None):
    """
    从指定索引开始获取数据集切片
    """
    if total_count is None:
        total_count = dataset.count()
    
    if start_index <= 0:
        print("Starting from the beginning of dataset")
        return dataset
    
    if start_index >= total_count:
        print(f"Start index {start_index} >= total count {total_count}, no data to process")
        return ray.data.from_items([])  # 返回空数据集
    
    remaining_count = total_count - start_index
    print(f"Skipping first {start_index} samples, processing remaining {remaining_count} samples")
    
    # 使用 skip 方法跳过已处理的数据
    return dataset.skip(start_index)


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
    ray_batch_size=None, enable_resume=False
):
    import datetime, gc, os
    os.makedirs(output_dir_path, exist_ok=True)

    total = dataset.count()
    
    # 断点续传逻辑
    start_index = 0
    if enable_resume:
        print("="*60)
        print("Resuming from last checkpoint")
        start_index = get_last_processed_index(output_dir_path, task)
        if start_index > 0:
            dataset = get_dataset_slice_from_index(dataset, start_index, total)
            # 重新计算剩余的数据量
            remaining_total = total - start_index
            print(f"🔄 Resuming from index {start_index}")
            print(f"📊 Remaining samples to process: {remaining_total}")
        else:
            remaining_total = total
            print(f"🆕 Starting fresh processing")
            print(f"📊 Total samples to process: {remaining_total}")
    else:
        remaining_total = total
        print(f"🚫 Resume disabled, processing all {total} samples")

    step = ray_batch_size # 给个上限，防爆内存
    
    # 调整checkpoint逻辑以考虑已处理的样本
    processed = start_index  # 从已处理的数量开始计算
    next_ckpt = ((start_index // checkpoint_interval) + 1) * checkpoint_interval
    
    buffer = []
    batch_idx = 0
    current_batch_processed = 0  # 当前会话处理的样本数

    print("="*60)
    print(f"Next checkpoint at: {next_ckpt} samples")
    print(f"Checkpoint interval: {checkpoint_interval}")
    

    for batch in dataset.iter_batches(batch_format="pandas",
                                      batch_size=step):
        print("="*60)
        print(f"Processing batch {batch_idx} of {total} samples")
        
        batch_idx += 1
        out_rows = list(processor(ray.data.from_pandas(batch)).iter_rows())
        buffer.extend(out_rows)
        processed += len(out_rows)
        current_batch_processed += len(out_rows)

        # flush 条件：到达checkpoint间隔或处理完所有数据
        if (processed >= next_ckpt and buffer) or processed == total:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir_path,
                                f"{task}_ckpt_{processed:07d}_{ts}")
            ray.data.from_items(buffer).repartition(1).write_parquet(path)
            print(f"💾 Checkpoint saved: {path}")
            print(f"📈 Progress: {processed}/{total} samples ({processed/total*100:.1f}%)")
            
            # 在保存checkpoint时显示最近的样本输出
            if show_sample_output and buffer:
                print(f"\n⏺ Recent samples (at checkpoint {processed}):")
                # 显示buffer中最后几个样本
                recent_samples = buffer[-max_sample_display:] if len(buffer) >= max_sample_display else buffer
                for i, r in enumerate(recent_samples):
                    print(f"  Sample {i+1}: {r.get('generated_text', '')[:80].replace('\n',' ')}")
                print("-" * 40)
            
            buffer.clear()
            next_ckpt += checkpoint_interval

        del batch, out_rows
        gc.collect()

    print("="*60)
    print(f"✅ Processing completed!")
    print(f"📊 Total processed in this session: {current_batch_processed}")
    print(f"📊 Total processed overall: {processed}/{total}")
    if start_index > 0:
        print(f"🔄 Resumed from index: {start_index}")

