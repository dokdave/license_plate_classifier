import torch, yaml
from torch.utils.data import DataLoader
from models.classifier import build_model
from data.dataset import LicensePlateDataset
from data.transforms import get_transforms
from utils.plots import plot_confusion

def validate(model_path, data_dir, model_name="resnet18", batch_size=32):
    dataset = LicensePlateDataset(data_dir, get_transforms("val"))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, 2)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    accuracy = correct / total if total > 0 else 0
    print(f'Validation accuracy: {accuracy:.4f}')
    plot_confusion(all_labels, all_preds, "runs/val_confusion.png")

if __name__ == "__main__":
    config = yaml.safe_load(open("config/config.yaml"))
    validate(
        model_path="runs/best_model.pth",
        data_dir=config["paths"]["val_dir"],
        model_name=config["training"]["model_name"]
    )
