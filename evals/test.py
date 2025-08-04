import torch, yaml, os
from models.classifier import build_model
from data.dataset import LicensePlateDataset
from torch.utils.data import DataLoader
from data.transforms import get_transforms
from utils.metrics import compute_metrics
from utils.plots import plot_confusion, plot_roc

def test():
    config = yaml.safe_load(open("config/config.yaml"))

    # ✅ num_classes = 1 for BCEWithLogitsLoss
    model = build_model(config["training"]["model_name"], 1, config["training"]["dropout"])
    model.load_state_dict(torch.load("runs/best_model.pth", map_location=torch.device("cpu")))
    model.eval()

    dataset = LicensePlateDataset(config["paths"]["test_dir"], get_transforms("val"))
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"])

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images).squeeze(1)  # shape: (B,)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds += preds.tolist()
            all_labels += labels.tolist()
            all_probs += probs.tolist()

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print(metrics)

    os.makedirs("runs", exist_ok=True)
    plot_confusion(all_labels, all_preds, "runs/test_confusion.png")
    plot_roc(all_labels, all_probs, "runs/test_roc.png")

if __name__ == "__main__":
    test()
