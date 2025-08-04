from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }