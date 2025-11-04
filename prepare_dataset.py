import kagglehub
import shutil
import os

# Download dataset to cache (default behavior)
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Define destination (current directory)
dest = os.path.join(os.getcwd(), "chest-xray-pneumonia")

# Copy from cache to current dir if not already there
if not os.path.exists(dest):
    shutil.copytree(path, dest)

print("Dataset copied to:", dest)
