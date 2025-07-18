import pandas as pd
from tqdm import tqdm
import re
import os


reason_itw = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet"

df = pd.read_parquet(reason_itw)

print(df.head())

print(df.columns)

print(df.shape)

print(df.info())