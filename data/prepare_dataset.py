import os, shutil, random
from pathlib import Path

def prepare_dataset(base_dir="data/images", output_root="data", seed=42):
    labels = ["readable", "unreadable"]
    splits = ["train", "val", "test"]
    ratios = [0.7, 0.2, 0.1]
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    random.seed(seed)

    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(output_root, split, label), exist_ok=True)

    for label in labels:
        files = []
        for ext in extensions:
            files.extend(Path(base_dir, label).glob(ext))
        random.shuffle(files)
        n = len(files)
        train_split = int(n * ratios[0])
        val_split = int(n * (ratios[0] + ratios[1]))

        split_files = {
            "train": files[:train_split],
            "val": files[train_split:val_split],
            "test": files[val_split:]
        }

        for split in splits:
            for f in split_files[split]:
                shutil.copy(f, Path(output_root, split, label, f.name))

if __name__ == "__main__":
    prepare_dataset()
