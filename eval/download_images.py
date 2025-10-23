import os
from datasets import load_dataset
from PIL import Image
import io
import requests
from tqdm import tqdm
from pathlib import Path


def download_coco_images(
    output_dir="/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/coco_images",
    split="test",
    max_samples=None,
    timeout=10
):
    """
    预下载 COCO Karpathy split 的所有图片到本地
    
    Args:
        output_dir: 图片保存目录
        split: 数据集分割 ("test", "validation")
        max_samples: 最多下载多少张（调试用）
        timeout: 下载超时时间（秒）
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 图片保存目录: {output_dir}")
    print(f"📥 加载 COCO Karpathy {split} split...")
    
    # 加载数据集
    ds = load_dataset("/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/coco-karpathy", split=split)
    
    print(f"✅ 加载了 {len(ds)} 张图片")
    print(f"开始下载...")
    
    failed_urls = []
    success_count = 0
    
    for idx, sample in enumerate(tqdm(ds)):
        if max_samples and idx >= max_samples:
            break
        
        if 'url' not in sample:
            continue
        
        url = sample['url']
        # 从 URL 提取图片 ID（通常是最后部分）
        image_id = sample.get('image_id', idx)
        
        # 保存路径
        save_path = output_path / f"{image_id}.jpg"
        
        # 如果已经存在，跳过
        if save_path.exists():
            success_count += 1
            continue
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # 打开图片并转换为 RGB（如果需要）
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            # 保存到本地
            image.save(save_path, format='JPEG')
            success_count += 1
            
        except Exception as e:
            failed_urls.append((image_id, url, str(e)))
            print(f"❌ 下载失败 (ID: {image_id}): {str(e)}")
    
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
    print("🚀 COCO 图片预下载工具")
    print("="*70)
    
    # 下载 test split
    print("\n1️⃣  下载 test split (5K 图片)...")
    test_success, test_failed = download_coco_images(
        split="test",
        max_samples=None,  # 下载全部
        timeout=10
    )
    
    # 下载 validation split
    print("\n2️⃣  下载 validation split (5K 图片)...")
    val_success, val_failed = download_coco_images(
        split="validation",
        max_samples=None,  # 下载全部
        timeout=10
    )
    
    print("\n" + "="*70)
    print("📊 下载总结:")
    print(f"  Test split: ✅ {test_success}, ❌ {test_failed}")
    print(f"  Val split:  ✅ {val_success}, ❌ {val_failed}")
    print("="*70)
