# evaluate_predictions.py

import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FILE = "results/inavit_predictions.json"

def evaluate():
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    for video_path, result in data.items():
        if result["mapped_label"] != "unknown":
            y_true.append(result["true_label"])
            y_pred.append(result["mapped_label"])

    print("📊 Classification Report (per-class metrics):")
    print(classification_report(y_true, y_pred, digits=3))

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Top-1 Accuracy: {acc:.3f}")

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    y_true, y_pred = evaluate()
    # plot_confusion_matrix(y_true, y_pred)
