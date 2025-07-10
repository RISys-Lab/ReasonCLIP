import pandas as pd
from tqdm import tqdm
import re
import os

reason_itw = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"
df1 = pd.read_parquet(reason_itw)

print(df1.head())
print(df1.columns)
print(f"数据形状: {df1.shape}")