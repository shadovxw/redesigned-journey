# prepare_data_split.py
import os
import random
import shutil
from pathlib import Path

# ----- USER CONFIG -----
SRC_DIR = "raw_dataset"   # source dir containing NORMAL/ and PNEUMONIA/ (or change to where you unzipped)
OUT_DIR = "data"
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 123
# ------------------------

random.seed(SEED)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def split_files_for_class(class_name):
    src_class_dir = Path(SRC_DIR) / class_name
    if not src_class_dir.exists():
        print(f"Warning: source class folder not found: {src_class_dir}")
        return []

    all_files = [f for f in src_class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    # Optional heuristic for patient-level grouping:
    # If filenames look like "person123_bla.jpg" then we can group by the prefix "person123".
    # The simple fallback here just randomly splits files individually.
    random.shuffle(all_files)
    n = len(all_files)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    return train_files, val_files, test_files

def copy_list(file_list, dest_dir):
    ensure_dir(dest_dir)
    for f in file_list:
        shutil.copy2(f, Path(dest_dir) / f.name)

def main():
    classes = ["NORMAL", "PNEUMONIA"]
    # create target dirs
    for split in ("train", "val", "test"):
        for cls in classes:
            ensure_dir(Path(OUT_DIR) / split / cls)

    summary = {}
    for cls in classes:
        res = split_files_for_class(cls)
        if not res:
            continue
        train_files, val_files, test_files = res
        copy_list(train_files, Path(OUT_DIR) / "train" / cls)
        copy_list(val_files,   Path(OUT_DIR) / "val"   / cls)
        copy_list(test_files,  Path(OUT_DIR) / "test"  / cls)
        summary[cls] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }

    print("Done. Summary:")
    for cls, counts in summary.items():
        print(f"  {cls}: train={counts['train']}, val={counts['val']}, test={counts['test']}")

if __name__ == "__main__":
    main()
