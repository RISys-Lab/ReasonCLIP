import pandas as pd
import os
import shutil
from tqdm import tqdm
from pathlib import Path

# Read parquet file
parquet_file = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/data/fesvhtr-CLIPReasonItw/llavacot_test_with_best_trp.parquet"
# parquet_file = "/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/llavacot_test_with_best_trp.parquet"
df = pd.read_parquet(parquet_file)

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())
for i in range(0, 10):
    print(df.iloc[i]['image_path'])

# Create base output directory
base_output_dir = "/leonardo_work/EUHPC_R04_192/fmohamma/CLIP-R/outputs/ReasonPro/llavacot_test_images"
os.makedirs(base_output_dir, exist_ok=True)
print(f"\nCreating base output directory: {base_output_dir}")

# Copy images with preserved directory structure
success_count = 0
error_count = 0
missing_count = 0

for i in tqdm(range(len(df)), desc="Copying images"):
    try:
        image_path = df.iloc[i]['image_path']
        
        # Check if source file exists
        if os.path.exists(image_path):
            # Extract path after 'Xkev-LLaVA-CoT-100k'
            if 'Xkev-LLaVA-CoT-100k' in image_path:
                # Split the path and find the part after Xkev-LLaVA-CoT-100k
                path_parts = image_path.split('Xkev-LLaVA-CoT-100k')
                if len(path_parts) > 1:
                    relative_path = path_parts[1].lstrip('/')  # Remove leading slash
                    
                    # Create the output path with preserved structure
                    output_path = os.path.join(base_output_dir, relative_path)
                    
                    # Create necessary directories
                    output_dir = os.path.dirname(output_path)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Handle duplicate filenames by adding index
                    if os.path.exists(output_path):
                        base_name, ext = os.path.splitext(output_path)
                        counter = 1
                        while os.path.exists(f"{base_name}_{counter}{ext}"):
                            counter += 1
                        output_path = f"{base_name}_{counter}{ext}"
                    
                    # Copy file
                    shutil.copy2(image_path, output_path)
                    success_count += 1
                else:
                    # Fallback: use original filename if path parsing fails
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(base_output_dir, filename)
                    shutil.copy2(image_path, output_path)
                    success_count += 1
            else:
                # If 'Xkev-LLaVA-CoT-100k' not found, use original filename
                filename = os.path.basename(image_path)
                output_path = os.path.join(base_output_dir, filename)
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
print(f"Images saved to: {base_output_dir}")

# Check output directory structure
def count_files_recursive(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

if os.path.exists(base_output_dir):
    total_copied = count_files_recursive(base_output_dir)
    print(f"Actual copied files: {total_copied}")
    
    # Show directory structure
    print("\nDirectory structure created:")
    for root, dirs, files in os.walk(base_output_dir):
        level = root.replace(base_output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files in each directory
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files") 