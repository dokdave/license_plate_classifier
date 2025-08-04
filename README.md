# License Plate Readability Classifier
A fast, lightweight deep learning model for classifying vehicle license plate images as readable or unreadable. Built with PyTorch and MobileNetV3, optimized for CPU inference. Includes training pipeline, evaluation metrics, false positive/negative analysis, and visualization tools.

## 🔧 Features

- MobileNetV3 backbone (`mobilenetv3_small_100`)
- Binary classification with `BCEWithLogitsLoss` and `pos_weight` support
- CPU-friendly inference (<15ms per image)
- Training/validation performance tracking
- Evaluation: F1, precision, recall, accuracy
- Visual reports: confusion matrix, ROC AUC
- Saves false positives and false negatives for analysis
- Configurable via `config.yaml`

- ## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/license-plate-classifier.git
cd license-plate-classifier

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Expected folder structure inside data/:
```
data/
├── train/
│   ├── readable/
│   └── unreadable/
├── val/
│   ├── readable/
│   └── unreadable/
├── test/
│   ├── readable/
│   └── unreadable/
```

- ## 🚀 Training
Edit hyperparameters in config/config.yaml if needed. Then run:
```
PYTHONPATH=$(pwd) python3 scripts/run_training.py
```

- ## 📊 Evaluation
To test the model and generate ROC and confusion matrix:
```
PYTHONPATH=$(pwd) python3 evals/test.py
```

- ## 🕵️‍♂️ Analyze False Predictions
Save images that were falsely predicted into data/falsenegative/:
```
PYTHONPATH=$(pwd) python3 scripts/false_negative.py
```

- ## ⚡ Measure Inference Time
```
PYTHONPATH=$(pwd) python3 evals/inference.py
```

- ## 📈 Example Outputs
```
✅ Best F1: 0.93+
📉 Inference Time: ~13.5 ms/image (CPU)
📊 ROC AUC, Confusion Matrix, Per-Class Recall
```
