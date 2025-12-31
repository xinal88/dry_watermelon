import os

# 🔥 SET ENV TRƯỚC KHI IMPORT kagglehub
os.environ["KAGGLEHUB_CACHE_DIR"] = r"F:\kagglehub_cache"

import kagglehub
import os
import shutil
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATASET_ID = "orvile/ravdess-dataset"
TARGET_DIR = r"F:\AI_project\dry_watermelon\data\ravdess"

os.makedirs(TARGET_DIR, exist_ok=True)

# =====================
# DOWNLOAD DATASET
# =====================
print("Downloading dataset from Kaggle...")
cached_path = kagglehub.dataset_download(DATASET_ID)
print("Cached path:", cached_path)

# =====================
# COPY WITH PROGRESS BAR
# =====================
def copy_with_progress(src, dst):
    files = []
    for root, _, filenames in os.walk(src):
        for f in filenames:
            files.append(os.path.join(root, f))

    with tqdm(total=len(files), desc="Copying RAVDESS", unit="file") as pbar:
        for file in files:
            rel = os.path.relpath(file, src)
            dest = os.path.join(dst, rel)

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(file, dest)
            pbar.update(1)

copy_with_progress(cached_path, TARGET_DIR)

print("✅ DONE")
