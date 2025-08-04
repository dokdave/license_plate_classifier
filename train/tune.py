import optuna, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from models.classifier import build_model
from data.dataset import LicensePlateDataset
from data.transforms import get_transforms
from train import evaluate

def objective(trial):
    model_name = trial.suggest_categorical("model_name", ["mobilenetv3_small_100"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    device = torch.device("cpu")
    train_tf, val_tf = get_transforms(224)
    train_ds = LicensePlateDataset("data/train", transform=train_tf)
    val_ds = LicensePlateDataset("data/val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = build_model(model_name, 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
    return val_f1

def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best trial:", study.best_trial.params)
    return study