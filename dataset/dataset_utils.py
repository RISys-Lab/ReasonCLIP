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
    通过检查checkpoint文件夹名中的数字来确定
    只检查文件夹格式的checkpoint
    """
    if not os.path.exists(output_dir_path):
        print(f"Output directory {output_dir_path} does not exist, starting from beginning")
        return 0
    
    # 查找所有checkpoint文件夹
    checkpoint_pattern = os.path.join(output_dir_path, f"{task}_ckpt_*")
    checkpoint_dirs = [d for d in glob.glob(checkpoint_pattern) if os.path.isdir(d)]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {output_dir_path}, starting from beginning")
        return 0
    
    # 从文件夹名中提取已处理的样本数量
    max_processed = 0
    latest_checkpoint = None
    
    for checkpoint_dir in checkpoint_dirs:
        dir_name = os.path.basename(checkpoint_dir)
        # 匹配格式: {task}_ckpt_{processed:07d}_{timestamp}
        match = re.search(rf'{re.escape(task)}_ckpt_(\d+)_', dir_name)
        if match:
            processed_count = int(match.group(1))
            if processed_count > max_processed:
                max_processed = processed_count
                latest_checkpoint = checkpoint_dir
    
    if max_processed > 0:
        print(f"Found latest checkpoint directory: {latest_checkpoint}")
        print(f"Last processed index: {max_processed}")
        return max_processed
    else:
        print("No valid checkpoint directories found, starting from beginning")
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

    
    _, tail = dataset.split_at_indices([start_index])
    print(f"Skipping first {start_index} samples, processing remaining {total_count - start_index} samples")
    return tail


def process_dataset_with_checkpoints_optimized_deprecated(
    dataset, processor, task,
    checkpoint_interval, output_dir_path,
    show_sample_output=True, max_sample_display=3,
    ray_batch_size=None, enable_resume=False
):
    import datetime, gc, os, math
    os.makedirs(output_dir_path, exist_ok=True)

    # -------- 1) 计算全量与断点 --------
    total_all = dataset.count()
    start_index = 0
    if enable_resume:
        print("=" * 60)
        print("Resuming from last checkpoint")
        start_index = get_last_processed_index(output_dir_path, task)
    else:
        print("🚫 Resume disabled")

    # 切掉已经处理的部分
    if start_index > 0:
        dataset = get_dataset_slice_from_index(dataset, start_index, total_all)
        print(f"🔄 Resuming from index {start_index}")
    else:
        print("🆕 Starting fresh")

    # 剩余要处理的数量 & 真正终点（全局索引）
    tail_count = dataset.count()
    end_index = start_index + tail_count

    print(f"📊 Remaining samples to process: {tail_count}")
    print(f"📊 Global end index: {end_index}/{total_all}")

    # change to gen_ds
    gen_ds = processor(dataset)
    # -------- 2) 初始化计数与 ckpt 位置 --------
    step = ray_batch_size  # None 或者整数
    processed = start_index            # 全局已处理计数
    next_ckpt = ((start_index // checkpoint_interval) + 1) * checkpoint_interval

    buffer = []
    batch_idx = 0
    current_session_processed = 0

    print("=" * 60)
    print(f"Next checkpoint at: {next_ckpt}")
    print(f"Checkpoint interval: {checkpoint_interval}")

    # 小工具：flush 写盘
    def flush_buffer():
        nonlocal buffer, processed
        if not buffer:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir_path, f"{task}_ckpt_{processed:07d}_{ts}")
        ray.data.from_items(buffer).repartition(1).write_parquet(path)
        print(f"💾 Checkpoint saved: {path}")
        pct = (processed / end_index) * 100 if end_index else 0
        print(f"📈 Progress: {processed}/{end_index} ({pct:.1f}%)")

        if show_sample_output:
            print("\n⏺ Recent samples:")
            recent = buffer[-max_sample_display:] if len(buffer) >= max_sample_display else buffer
            for i, r in enumerate(recent, 1):
                txt = r.get('generated_text', '')
                print(f"  Sample {i}: {txt[:80].replace('\\n',' ')}")
            print("-" * 40)

        buffer.clear()

    # -------- 3) 主循环 --------
    # change to gen_ds
    for batch in gen_ds.iter_batches(batch_format="pandas", batch_size=step):
        print("=" * 60)
        print(f"Processing batch {batch_idx} "
              f"({processed - start_index}/{tail_count} this run)")
        batch_idx += 1

        # change to gen_ds
        # out_rows = list(processor(ray.data.from_pandas(batch)).iter_rows())
        out_rows = batch.to_dict(orient="records")
        buffer.extend(out_rows)
        processed += len(out_rows)
        current_session_processed += len(out_rows)

        # 触发条件：到达 ckpt 或到达真正终点
        if (processed >= next_ckpt and buffer) or processed == end_index:
            flush_buffer()
            next_ckpt += checkpoint_interval

        del batch, out_rows
        gc.collect()

    # -------- 4) 兜底 flush --------
    if buffer:
        print("🧹 Final flush...")
        flush_buffer()

    # -------- 5) 收尾打印 --------
    print("=" * 60)
    print("✅ Processing completed!")
    print(f"📊 Total processed in this session: {current_session_processed}")
    print(f"📊 Total processed overall: {processed}/{end_index}")
    if start_index > 0:
        print(f"🔄 Resumed from index: {start_index}")

def process_dataset_with_checkpoints_optimized(
    dataset, processor, task,
    checkpoint_interval, output_dir_path,
    show_sample_output=True, max_sample_display=3,
    ray_batch_size=None, enable_resume=False
):
    import datetime, gc, os, math
    import pandas as pd
    os.makedirs(output_dir_path, exist_ok=True)

    # -------- 1) 计算全量与断点 --------
    total_all = dataset.count()
    start_index = 0
    if enable_resume:
        print("=" * 60)
        print("Resuming from last checkpoint")
        start_index = get_last_processed_index(output_dir_path, task)
    else:
        print("🚫 Resume disabled")

    # 切掉已经处理的部分
    if start_index > 0:
        dataset = get_dataset_slice_from_index(dataset, start_index, total_all)
        print(f"🔄 Resuming from index {start_index}")
    else:
        print("🆕 Starting fresh")

    # 剩余要处理的数量 & 真正终点（全局索引）
    tail_count = dataset.count()
    end_index = start_index + tail_count

    print(f"📊 Remaining samples to process: {tail_count}")
    print(f"📊 Global end index: {end_index}/{total_all}")

    # change to gen_ds
    gen_ds = processor(dataset)
    # -------- 2) 初始化计数与 ckpt 位置 --------
    step = ray_batch_size  # None 或者整数
    processed = start_index            # 全局已处理计数
    next_ckpt = start_index + checkpoint_interval  # 修正边界初始化

    buffer = []
    batch_idx = 0
    current_session_processed = 0

    print("=" * 60)
    print(f"Next checkpoint at: {next_ckpt}")
    print(f"Checkpoint interval: {checkpoint_interval}")

    # 小工具：flush 写盘
    def flush_buffer():
        nonlocal buffer, processed
        if not buffer:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir_path, f"{task}_ckpt_{processed:07d}_{ts}")

        # 去重，确保同一个 checkpoint 内不会重复写入相同 id
        df = pd.DataFrame(buffer)
        df = df.drop_duplicates(subset=["id"])
        ray.data.from_pandas(df).repartition(1).write_parquet(path)

        print(f"💾 Checkpoint saved: {path}")
        pct = (processed / end_index) * 100 if end_index else 0
        print(f"📈 Progress: {processed}/{end_index} ({pct:.1f}%)")

        if show_sample_output:
            print("\n⏺ Recent samples:")
            recent = buffer[-max_sample_display:] if len(buffer) >= max_sample_display else buffer
            for i, r in enumerate(recent, 1):
                txt = r.get('generated_text', '')
                print(f"  Sample {i}: {txt[:80].replace('\\n',' ')}")
            print("-" * 40)

        buffer.clear()

    # -------- 3) 主循环 --------
    for batch in gen_ds.iter_batches(batch_format="pandas", batch_size=step):
        print("=" * 60)
        print(f"Processing batch {batch_idx} "
              f"({processed - start_index}/{tail_count} this run)")
        batch_idx += 1

        out_rows = batch.to_dict(orient="records")
        buffer.extend(out_rows)
        processed += len(out_rows)
        current_session_processed += len(out_rows)

        # 触发条件：严格超过 ckpt（避免边界重复），并支持一次跨多 ckpt
        while processed > next_ckpt:
            flush_buffer()
            next_ckpt += checkpoint_interval

        del batch, out_rows
        gc.collect()

    # -------- 4) 兜底 flush --------
    if buffer:
        print("🧹 Final flush...")
        flush_buffer()

    # -------- 5) 收尾打印 --------
    print("=" * 60)
    print("✅ Processing completed!")
    print(f"📊 Total processed in this session: {current_session_processed}")
    print(f"📊 Total processed overall: {processed}/{end_index}")
    if start_index > 0:
        print(f"🔄 Resumed from index: {start_index}")


