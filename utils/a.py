import pandas as pd
import re

def clean_text_block(text_block):
    """去掉编号、换行和多余空格，将3段文字变成list"""
    parts = re.split(r'\n+', str(text_block).strip())
    cleaned = []
    for p in parts:
        p = re.sub(r'^\s*\d+[\.\)]\s*', '', p.strip())
        if p:
            cleaned.append(p)
    return cleaned

def pad_to_three(lst):
    """若不足3个元素，用第一个元素补齐"""
    if not lst:
        return [""] * 3
    while len(lst) < 3:
        lst.append(lst[0])
    return lst[:3]

# === 处理 chunk03–05 ===
for i in range(3, 6):
    tb_path = f"/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_tb/combined/cc12m_tb_chunk_0{i}.parquet"
    trl_path = f"/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/combined_unclassified/cc12m_trl_chunk0{i}.parquet"
    out_path = f"/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonLite/cc12m_trl/final_unclassified/cc12m_tb_trl_chunk0{i}.parquet"

    print(f"\n🔹 Processing chunk {i}...")

    tb_df = pd.read_parquet(tb_path)
    trl_df = pd.read_parquet(trl_path)

    tb_df["tb"] = tb_df["tb"].apply(clean_text_block)
    trl_df["trl"] = trl_df["generated_text"].apply(clean_text_block)

    merged = pd.merge(
        tb_df[["id", "image_path", "tb"]],
        trl_df[["id", "trl"]],
        on="id",
        how="inner"
    )

    # === 统计补齐前 ===
    tb_len3 = merged["tb"].apply(len).eq(3).sum()
    trl_len3 = merged["trl"].apply(len).eq(3).sum()
    total = len(merged)

    # === 补齐 ===
    merged["tb"] = merged["tb"].apply(pad_to_three)
    merged["trl"] = merged["trl"].apply(pad_to_three)

    # === 保存 ===
    merged.to_parquet(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print(f"📊 Total samples: {total}")
    print(f"   TB length=3:  {tb_len3} ({tb_len3/total:.2%})")
    print(f"   TRL length=3: {trl_len3} ({trl_len3/total:.2%})")
    print(f"   TB padded:    {total - tb_len3}")
    print(f"   TRL padded:   {total - trl_len3}")