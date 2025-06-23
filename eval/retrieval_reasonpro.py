import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class TRPRetrievalDataset(Dataset):
    """把原始 {image_path, trp: [trp1,trp2,trp3]} 展平成单对一的 (img_path, trp)"""
    def __init__(self, records, processor):
        self.processor = processor
        self.pairs = []
        for r_idx, r in enumerate(records):
            for trp in r["trp"]:
                # 记录 (图片索引, trp 文本)
                self.pairs.append((r["image_path"], trp, r_idx))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img_path, trp, img_idx = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        return img, trp, img_idx, idx

def collate_fn(batch, processor):
    imgs, trps, img_idxs, pair_idxs = zip(*batch)
    image_inputs = processor(images=list(imgs), return_tensors="pt")
    text_inputs  = processor(text=list(trps),
                             return_tensors="pt",
                             padding=True, truncation=True, max_length=77)
    return image_inputs, text_inputs, torch.tensor(img_idxs), torch.tensor(pair_idxs)

def evaluate_bi_directional(records,
                             model_id="openai/clip-vit-base-patch32",
                             batch_size=32,
                             device="cuda"):
    # 1️⃣ 加载模型
    model     = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    # 2️⃣ 准备 Dataset + DataLoader
    ds        = TRPRetrievalDataset(records, processor)
    loader    = DataLoader(ds,
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=lambda b: collate_fn(b, processor),
                           num_workers=4)

    # 3️⃣ 先算出所有 **唯一** 图片的特征（N 张图）
    unique_paths = [r["image_path"] for r in records]
    all_img_feats = []
    for i in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inp  = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
        all_img_feats.append(f.cpu())
    all_img_feats = torch.cat(all_img_feats, dim=0)  # [N, D]

    # 4️⃣ 再算出所有 TRP 文本特征 (3N 条)
    all_trp_feats = []
    pair2img       = []  # 记录每条 trp 属于哪张图
    for image_inputs, text_inputs, img_idxs, pair_idxs in loader:
        text_inputs = {k:v.to(device) for k,v in text_inputs.items()}
        with torch.no_grad():
            f = model.get_text_features(**text_inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        all_trp_feats.append(f.cpu())
        pair2img.append(img_idxs)
    all_trp_feats = torch.cat(all_trp_feats, dim=0)     # [3N, D]
    pair2img      = torch.cat(pair2img, dim=0).numpy()   # [3N]

    # 5️⃣ 计算相似度矩阵
    #    S_trp2img: [3N, N]   trp→图  
    #    S_img2trp: [N, 3N]   图→trp
    S_trp2img = all_trp_feats @ all_img_feats.T
    S_img2trp = S_trp2img.T

    # 6️⃣ 统计 TRP→Image 的 R@k
    trp_ranks = []
    for q in range(S_trp2img.shape[0]):
        sims  = S_trp2img[q]                        # [N]
        order = torch.argsort(sims, descending=True)
        gt    = pair2img[q]                         # 真实图索引
        rank  = (order == gt).nonzero()[0].item() + 1
        trp_ranks.append(rank)
    trp_ranks = np.array(trp_ranks)

    # 7️⃣ 统计 Image→TRP 的 R@k
    img_ranks = []
    # 每张图对应 3 条 trp，在 pair2img 中以 img_idx 匹配
    # 找到所有对应该图的 trp 的全索引列表
    img2pairs = {i: np.where(pair2img == i)[0].tolist() for i in range(len(unique_paths))}
    for img_i in range(S_img2trp.shape[0]):
        sims  = S_img2trp[img_i]                   # [3N]
        order = torch.argsort(sims, descending=True)
        gts   = img2pairs[img_i]                   # 该图对应的 trp 索引列表
        # 取最先检索到的那个 trp 的 rank（取 min）
        ranks = [(order == gt).nonzero()[0].item() + 1 for gt in gts]
        img_ranks.append(min(ranks))
    img_ranks = np.array(img_ranks)

    # 8️⃣ 打印
    def print_stats(name, ranks):
        print(f"── {name} R@1:  {np.mean(ranks<=1)*100:.2f}%")
        print(f"   {name} R@5:  {np.mean(ranks<=5)*100:.2f}%")
        print(f"   {name} R@10: {np.mean(ranks<=10)*100:.2f}%")
        print(f"   {name} MeanRank: {ranks.mean():.2f}")
        print(f"   {name} MedRank:  {np.median(ranks):.2f}\n")

    print_stats("TRP→Image", trp_ranks)
    print_stats("Image→TRP", img_ranks)

    return {
        "trp2img_ranks": trp_ranks,
        "img2trp_ranks": img_ranks,
    }

# ===== 调用示例 =====
if __name__ == "__main__":
    # records = load_your_dataset()  # 每条 record: {"image_path":..., "trp":[...]}
    metrics = evaluate_bi_directional(
        records,
        model_id="openai/clip-vit-large-patch32",
        batch_size=32,
        device="cuda"
    )
