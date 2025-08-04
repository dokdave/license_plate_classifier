import yaml

def load_config_with_override(trial):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["training"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config["training"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    config["training"]["model_name"] = trial.suggest_categorical("model_name", ["resnet18", "resnet34", "efficientnet_b0"])

    return config
