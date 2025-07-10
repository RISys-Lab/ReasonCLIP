from datasets import load_dataset

ds = load_dataset("Xkev/LLaVA-CoT-100k")
print(f"Dataset structure: {ds}")
print(f"Column names: {ds['train'].column_names}")