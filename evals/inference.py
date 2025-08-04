import time
import torch
from torch.utils.data import DataLoader
from data.dataset import LicensePlateDataset
from data.transforms import get_transforms
from models.classifier import build_model
import yaml
import os

def measure_inference_time(config_path="config/config.yaml", mode="val"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Dataloader
    data_dir = config["paths"][f"{mode}_dir"]
    dataset = LicensePlateDataset(data_dir, transform=get_transforms("val"))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load model
    model = build_model(
        model_name=config["training"]["model_name"],
        num_classes=1,
        dropout=config["training"]["dropout"]
    )
    model.load_state_dict(torch.load(os.path.join(config["paths"]["output_dir"], "best_model.pth"), map_location=device))
    model.to(device)
    model.eval()

    # Take one sample
    images, _ = next(iter(dataloader))
    images = images.to(device)

    # Warm-up GPU (optional)
    for _ in range(5):
        _ = model(images)

    # Measure inference time
    with torch.no_grad():
        start = time.time()
        _ = model(images)
        end = time.time()

    print(f"\n✅ Inference time for 1 image: {(end - start)*1000:.2f} ms")

if __name__ == "__main__":
    measure_inference_time()

