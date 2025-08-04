import os
import sys
import yaml
from train.train import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_config(config):
    try:
        training = config["training"]
        assert isinstance(training["model_name"], str)
        assert isinstance(training["batch_size"], int)
        assert isinstance(training["epochs"], int)
        assert isinstance(training["learning_rate"], float)
        assert isinstance(training["weight_decay"], float)
        assert isinstance(training["dropout"], float)
    except (AssertionError, KeyError, TypeError) as e:
        raise ValueError("❌ Invalid config.yaml: проверь типы данных в разделе 'training'") from e

if __name__ == "__main__":
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    train(config)
