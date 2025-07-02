from huggingface_hub import HfApi
import os

api = HfApi()
for file in os.listdir("/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo"):
    if file.endswith(".parquet"):
        api.upload_file(
            path_in_repo=file, 
            path_or_fileobj=f"/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonPro200K-Demo/{file}",  
            repo_id="fesvhtr/CLIPReasonPro200K", 
            repo_type="dataset" 
        )

# from huggingface_hub import hf_hub_download

# # 设定仓库ID
# repo_id = "Nicous/Hand-ICL"

# # 远程仓库中的文件路径
# path_in_repo = "fulldata_2.tar.gz"

# # 指定本地存储路径（可选）
# local_file = hf_hub_download(
#     repo_id=repo_id, 
#     filename=path_in_repo, 
#     repo_type="dataset",
#     local_dir="/home/muzammal/Projects/CLIP-R/data/"
# )

# print(f"文件已下载到: {local_file}")


# import os
# from huggingface_hub import HfApi

# # Initialize the Hugging Face API
# api = HfApi()

# # Repository information
# repo_id = "TRIG-bench/TRIG"
# repo_type = "dataset"

# # Directory containing ZIP files to upload
# zip_directory = "/home/muzammal/Projects/TRIG/data/output/p2p/"

# # Path prefix in the repository
# repo_prefix = "output/p2p/"

# # Iterate through all files in the directory
# for filename in os.listdir(zip_directory):
#     if filename.endswith('.zip'):
#         # Local file path
#         local_file_path = os.path.join(zip_directory, filename)
        
#         # Path in the repository
#         path_in_repo = os.path.join(repo_prefix, filename)
        
#         print(f"Uploading {filename} to {path_in_repo}...")
        
#         # Upload the file
#         api.upload_file(
#             path_in_repo=path_in_repo,
#             path_or_fileobj=local_file_path,
#             repo_id=repo_id,
#             repo_type=repo_type
#         )
        
#         print(f"Successfully uploaded {filename}")

# print("All ZIP files have been uploaded.")