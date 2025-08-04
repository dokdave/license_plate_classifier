import torch, os
from torch.utils.data import DataLoader
from data.dataset import LicensePlateDataset
from models.classifier import build_model
from data.transforms import get_transforms
from utils.metrics import compute_metrics
from utils.plots import plot_confusion, plot_roc, plot_loss_curves

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        labels = labels.float()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (outputs > 0).long()
        total += labels.size(0)
        correct += (preds == labels.long()).sum().item()
        total_loss += loss.item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def validate(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.float()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.4).long()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            total_loss += loss.item()

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["y_true"] = all_labels
    metrics["y_pred"] = all_preds
    metrics["y_prob"] = all_probs

    return total_loss / len(dataloader), metrics

def train(config):
    train_ds = LicensePlateDataset(config["paths"]["train_dir"], get_transforms("train"))
    val_ds = LicensePlateDataset(config["paths"]["val_dir"], get_transforms("val"))
    train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    model = build_model(config["training"]["model_name"], num_classes=1, dropout=config["training"]["dropout"])
    
    pos_weight = torch.tensor([2])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])

    best_f1 = 0.0
    train_losses, val_losses = [], []
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer)
        val_loss, metrics = validate(model, val_dl, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Train Loss {train_loss:.4f}, Val F1 {metrics['f1']:.4f} , Val Recall {metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs(config["paths"]["output_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config["paths"]["output_dir"], "best_model.pth"))
            plot_confusion(metrics["y_true"], metrics["y_pred"],
                           os.path.join(config["paths"]["output_dir"], "confusion.png"))
            plot_roc(metrics["y_true"], metrics["y_prob"],
                     os.path.join(config["paths"]["output_dir"], "roc.png"))

    print("Training complete. Best F1 score:", best_f1)
    plot_loss_curves(train_losses, val_losses,
                    os.path.join(config["paths"]["output_dir"], "loss_curve.png"))
