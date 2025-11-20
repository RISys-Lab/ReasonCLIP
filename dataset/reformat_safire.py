from datasets import load_dataset

raw_ds = load_dataset("fesvhtr/iferniu", split="train")

for row in raw_ds:
    print(row["caption"])
    break