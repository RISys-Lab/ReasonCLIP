import pandas as pd
from tqdm import tqdm
import re

reason_pro_file_trp = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/llavacot_test_with_best_trp.parquet"

df_trp = pd.read_parquet(reason_pro_file_trp)

print(df_trp.head())
print(df_trp.columns)