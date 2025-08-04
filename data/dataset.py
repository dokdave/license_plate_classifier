from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os

class LicensePlateDataset(Dataset):
    def __init__(self, folder, transform=None, return_paths=False):
        self.samples = []
        self.transform = transform
        self.return_paths = return_paths 

        for label, cls in enumerate(["readable", "unreadable"]):
            cls_dir = os.path.join(folder, cls)
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(".jpg"):
                    continue
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return image, label, path 
        return image, label  
