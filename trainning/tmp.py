from datasets import load_dataset

raw_ds = load_dataset("fesvhtr/iferniu")
print(f"Dataset structure: {raw_ds}")
print(f"Column names: {raw_ds['train'].column_names}")

# Display the first few entries in the training set
print("First few entries in the training set:")
for i in range(5):
    print(raw_ds['train'][i])