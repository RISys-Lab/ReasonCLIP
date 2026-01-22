import os, json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

CKPT = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/siglip-so400m-patch14-384"

VAL_IMG_DIR = os.environ.get("COCO_VAL_IMG_DIR", "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/val2017")
CAP_JSON    = os.environ.get("COCO_CAP_JSON",    "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/annotations/captions_val2017.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_pairs(captions_json_path):
    coco = json.load(open(captions_json_path, "r"))
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # 固定图像顺序（val2017 是 5k）
    images = []
    img2text = []
    img_index = {}

    for i, (image_id, fn) in enumerate(id2file.items()):
        images.append((image_id, fn))
        img_index[image_id] = i
        img2text.append([])

    all_texts = []
    text2img = []

    # 每张图通常有 5 条 caption
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        cap = ann["caption"]
        i = img_index[image_id]
        img2text[i].append(len(all_texts))
        all_texts.append(cap)
        text2img.append(i)

    return images, all_texts, text2img, img2text

@torch.no_grad()
def encode_images(model, processor, image_paths, bs=32):
    feats=[]
    for i in tqdm(range(0, len(image_paths), bs), desc="Encode images"):
        batch=[Image.open(p).convert("RGB") for p in image_paths[i:i+bs]]
        inp=processor(images=batch, return_tensors="pt").to(device)
        x=model.get_image_features(**inp)
        x=F.normalize(x.float(), dim=-1)
        feats.append(x.cpu())
    return torch.cat(feats, 0)

@torch.no_grad()
def encode_texts(model, processor, texts, bs=256):
    feats=[]
    for i in tqdm(range(0, len(texts), bs), desc="Encode texts"):
        batch=[t.lower() for t in texts[i:i+bs]]  # IMPORTANT: lower-case
        inp=processor(
            text=batch,
            padding="max_length",  # IMPORTANT
            max_length=64,         # IMPORTANT
            truncation=True,
            return_tensors="pt",
        ).to(device)
        x=model.get_text_features(**inp)
        x=F.normalize(x.float(), dim=-1)
        feats.append(x.cpu())
    return torch.cat(feats, 0)

def t2i_recall(text_feats, image_feats, text2img):
    correct_1, correct_5, correct_10 = 0, 0, 0
    bs=1024
    for i in tqdm(range(0, text_feats.size(0), bs), desc="T->I Recall"):
        sims=text_feats[i:i+bs] @ image_feats.T
        topk=sims.topk(10, dim=1)[1]  # [batch, 10]
        gt=torch.tensor(text2img[i:i+bs]).unsqueeze(1)  # [batch, 1]
        
        correct_1 += (topk[:, :1]==gt).any(1).sum().item()
        correct_5 += (topk[:, :5]==gt).any(1).sum().item()
        correct_10 += (topk[:, :10]==gt).any(1).sum().item()
    
    total = text_feats.size(0)
    return correct_1/total, correct_5/total, correct_10/total

def i2t_recall(text_feats, image_feats, img2text):
    correct_1, correct_5, correct_10 = 0, 0, 0
    bs=256
    for i in tqdm(range(0, image_feats.size(0), bs), desc="I->T Recall"):
        sims=image_feats[i:i+bs] @ text_feats.T
        topk=sims.topk(10, dim=1)[1].tolist()  # [batch, 10]
        
        for j, preds in enumerate(topk):
            gt_set = set(img2text[i+j])
            if preds[0] in gt_set:
                correct_1 += 1
            if any(p in gt_set for p in preds[:5]):
                correct_5 += 1
            if any(p in gt_set for p in preds[:10]):
                correct_10 += 1
    
    total = image_feats.size(0)
    return correct_1/total, correct_5/total, correct_10/total

def main():
    model = AutoModel.from_pretrained(CKPT).to(device).eval()
    processor = AutoProcessor.from_pretrained(CKPT)

    images, texts, text2img, img2text = build_pairs(CAP_JSON)
    image_paths = [os.path.join(VAL_IMG_DIR, fn) for _, fn in images]

    print(f"#images={len(image_paths)}  #texts={len(texts)} (expect ~5k and ~25k)")

    img_f = encode_images(model, processor, image_paths, bs=32)
    txt_f = encode_texts(model, processor, texts, bs=256)

    t2i_r1, t2i_r5, t2i_r10 = t2i_recall(txt_f, img_f, text2img)
    i2t_r1, i2t_r5, i2t_r10 = i2t_recall(txt_f, img_f, img2text)

    print("\n=== COCO 2017 val retrieval ===")
    print(f"T->I R@1:  {t2i_r1*100:.2f}")
    print(f"T->I R@5:  {t2i_r5*100:.2f}")
    print(f"T->I R@10: {t2i_r10*100:.2f}")
    print(f"I->T R@1:  {i2t_r1*100:.2f}")
    print(f"I->T R@5:  {i2t_r5*100:.2f}")
    print(f"I->T R@10: {i2t_r10*100:.2f}")

if __name__ == "__main__":
    main()