import os, yaml
import shutil
import torch
from torch.utils.data import DataLoader
from data.dataset import LicensePlateDataset
from data.transforms import get_transforms
from models.classifier import build_model
from tqdm import tqdm
from PIL import Image

def save_false_negatives(config):
    os.makedirs("data/false_negative", exist_ok=True)

    # Загрузка модели
    model = build_model(config["training"]["model_name"], num_classes=1, dropout=config["training"]["dropout"])
    model.load_state_dict(torch.load(os.path.join(config["paths"]["output_dir"], "best_model.pth")))
    model.eval()

    # Подготовка данных
    dataset = LicensePlateDataset(config["paths"]["val_dir"], get_transforms("val"), return_paths=True)
    dataloader = DataLoader(dataset, batch_size=1)

    # Обход и проверка ошибок
    for image, label, path in tqdm(dataloader, desc="Checking for false negatives"):
        image = image
        label = label.float()

        with torch.no_grad():
            output = model(image).squeeze(1)
            prob = torch.sigmoid(output)
            pred = (prob > 0.4).long()

        if label.item() == 1 and pred.item() == 0:
            filename = os.path.basename(path[0])
            shutil.copy(path[0], os.path.join("data/false_negative", filename))

    print("False negatives saved to: data/false_negative/")

if __name__ == "__main__":
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    save_false_negatives(config)