import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

PREDICTIONS_FILE = "results/inavit_predictions.json"
GROUND_TRUTH_FILE = "EPIC_100_test_labels.csv"  # Must contain 'segment_id', 'action_id'

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['segment_id'], df['action_id']))

def evaluate():
    gt_dict = load_ground_truth(GROUND_TRUTH_FILE)

    with open(PREDICTIONS_FILE, "r") as f:
        preds = json.load(f)

    y_true = []
    y_pred = []

    for item in preds:
        seg_id = item["segment_id"]
        pred = item["pred_label"]
        if seg_id not in gt_dict:
            print(f"[WARN] Ground truth not found for {seg_id}")
            continue
        y_true.append(gt_dict[seg_id])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Top-1 Accuracy: {acc:.3f}")

    print("\n📊 Classification Report (per-class):")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, max_classes=30):
    from collections import Counter
    most_common = [c for c, _ in Counter(y_true).most_common(max_classes)]

    y_true_filt = [y for y in y_true if y in most_common]
    y_pred_filt = [p for y, p in zip(y_true, y_pred) if y in most_common]

    labels = sorted(set(y_true_filt + y_pred_filt))
    cm = confusion_matrix(y_true_filt, y_pred_filt, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Top {max_classes} classes)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    y_true, y_pred = evaluate()
    # plot_confusion_matrix(y_true, y_pred)
