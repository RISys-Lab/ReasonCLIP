import pandas as pd

reason_pro_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/llavacot_combined.parquet"

df = pd.read_parquet(reason_pro_file)

print(len(df))
for i in range(1, 5):
    print(df.iloc[i])




