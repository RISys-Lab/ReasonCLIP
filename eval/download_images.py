import os
from datasets import load_dataset
from PIL import Image
import io
import requests
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# 全局锁用于线程安全的计数
lock = threading.Lock()
success_count = 0
failed_urls = []


def download_single_image(args):
    """下载单张图片"""
    global success_count, failed_urls
    
    image_id, url, save_path, timeout = args
    
    # 如果已经存在，跳过
    if Path(save_path).exists():
        with lock:
            success_count += 1
        return True, image_id
    
    try:
        # 使用 stream=True 避免一次性加载整个图片到内存
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            # 直接将内容写入文件，不做转换
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        with lock:
            success_count += 1
        return True, image_id
        
    except Exception as e:
        with lock:
            failed_urls.append((image_id, url, str(e)))
        return False, image_id


def download_coco_images(
    output_dir="/home/muzammal/Projects/CLIP-R/data/coco_images",  # 保存到本地 data 目录
    split="test",
    max_samples=None,
    timeout=10,
    num_threads=16  # 👈 并发线程数
):
    """
    🚀 多线程预下载 COCO Karpathy split 的所有图片到本地
    
    Args:
        output_dir: 图片保存目录
        split: 数据集分割 ("test", "validation")
        max_samples: 最多下载多少张（调试用）
        timeout: 下载超时时间（秒）
        num_threads: 并发线程数（建议 8-32，根据网速调整）
    """
    global success_count, failed_urls
    success_count = 0
    failed_urls = []
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 图片保存目录: {output_dir}")
    print(f"🚀 并发线程数: {num_threads}")
    print(f"📥 加载 COCO Karpathy {split} split...")
    
    # 加载数据集
    ds = load_dataset("yerevann/coco-karpathy", split=split)
    
    print(f"✅ 加载了 {len(ds)} 张图片")
    print(f"⏳ 准备下载任务...")
    
    # 准备下载任务列表
    download_tasks = []
    for idx, sample in enumerate(ds):
        if max_samples and idx >= max_samples:
            break
        
        if 'url' not in sample:
            continue
        
        url = sample['url']
        # ✅ 直接从 URL 提取文件名（比如 COCO_train2014_000000057870.jpg）
        filename = os.path.basename(url)
        save_path = str(output_path / filename)
        
        download_tasks.append((filename, url, save_path, timeout))
    
    print(f"📦 共 {len(download_tasks)} 张图片待下载\n")
    
    # 多线程并发下载
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用 tqdm 显示进度
        futures = {
            executor.submit(download_single_image, task): task[0] 
            for task in download_tasks
        }
        
        with tqdm(as_completed(futures), total=len(futures), desc="下载进度") as pbar:
            for future in pbar:
                try:
                    success, image_id = future.result()
                except Exception as e:
                    print(f"❌ 任务异常: {e}")
    
    print("\n" + "="*70)
    print(f"✅ 成功下载: {success_count} 张")
    print(f"❌ 失败: {len(failed_urls)} 张")
    print("="*70)
    
    # 保存失败列表供参考
    if failed_urls:
        failed_file = output_path / "failed_downloads.txt"
        with open(failed_file, 'w') as f:
            for image_id, url, error in failed_urls:
                f.write(f"ID: {image_id}\nURL: {url}\nError: {error}\n\n")
        print(f"❌ 失败详情已保存到: {failed_file}")
    
    return success_count, len(failed_urls)


if __name__ == "__main__":
    print("🚀 COCO 图片预下载工具 (多线程加速版)")
    print("="*70)
    
    # 下载 test split
    print("\n1️⃣  下载 test split (5K 图片)...")
    test_success, test_failed = download_coco_images(
        split="test",
        max_samples=None,  # 下载全部
        timeout=5,  # 降低超时时间，快速跳过慢的连接
        num_threads=32  # 增加到 32 线程
    )
    
    # 下载 validation split
    print("\n2️⃣  下载 validation split (5K 图片)...")
    val_success, val_failed = download_coco_images(
        split="validation",
        max_samples=None,  # 下载全部
        timeout=5,  # 降低超时时间，快速跳过慢的连接
        num_threads=32  # 增加到 32 线程
    )
    
    print("\n" + "="*70)
    print("📊 下载总结:")
    print(f"  Test split: ✅ {test_success}, ❌ {test_failed}")
    print(f"  Val split:  ✅ {val_success}, ❌ {val_failed}")
    print(f"  总计: ✅ {test_success + val_success}, ❌ {test_failed + val_failed}")
    print("="*70)
