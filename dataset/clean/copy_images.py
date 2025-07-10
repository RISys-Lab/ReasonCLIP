import pandas as pd
import os
import shutil
from tqdm import tqdm
from pathlib import Path

# Read parquet file
parquet_file = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/fesvhtr-CLIPReasonItw/llavacot_test_with_best_trp.parquet"
df = pd.read_parquet(parquet_file)

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# Create output directory
output_dir = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/llavacot_test_images"
os.makedirs(output_dir, exist_ok=True)
print(f"\nCreating output directory: {output_dir}")

# Copy images
success_count = 0
error_count = 0
missing_count = 0

for i in tqdm(range(len(df)), desc="Copying images"):
    try:
        image_path = df.iloc[i]['image_path']
        
        # Check if source file exists
        if os.path.exists(image_path):
            # Get filename
            filename = os.path.basename(image_path)
            
            # If filename exists, add index
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, filename)
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                counter += 1
            
            # Copy file
            shutil.copy2(image_path, output_path)
            success_count += 1
            
        else:
            missing_count += 1
            if missing_count <= 10:  # Only print first 10 missing files
                print(f"File not found: {image_path}")
            
    except Exception as e:
        error_count += 1
        if error_count <= 10:  # Only print first 10 errors
            print(f"Copy failed {i}: {str(e)}")
            print(f"Image path: {df.iloc[i]['image_path']}")

# Print statistics
print(f"\nCopy completed!")
print(f"Total images: {len(df)}")
print(f"Successfully copied: {success_count}")
print(f"Missing files: {missing_count}")
print(f"Copy errors: {error_count}")
print(f"Success rate: {success_count/len(df)*100:.2f}%")
print(f"Images saved to: {output_dir}")

# Check output directory
copied_files = os.listdir(output_dir)
print(f"Actual copied files: {len(copied_files)}")
if len(copied_files) > 0:
    print("First few copied files:")
    for i, filename in enumerate(copied_files[:5]):
        print(f"  {filename}") 