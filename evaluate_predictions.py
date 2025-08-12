import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

PREDICTIONS_FILE = "results/inavit_predictions.json"
# Ground truth CSV must contain one of:
#  - ID column: 'narration_id' (preferred) or 'segment_id'
#  - Label column: 'action_id' (preferred) or 'action'
GROUND_TRUTH_FILE = "EPIC_100_test_labels.csv"

def _pick_col(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Ground truth CSV must contain a {kind} column in {candidates}")

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    id_col = _pick_col(df, ["narration_id", "segment_id"], "segment-id")
    y_col  = _pick_col(df, ["action_id", "action"], "action label")

    # Keep only needed columns
    df = df[[id_col, y_col]].dropna()
    # Cast labels to int if possible
    try:
        df[y_col] = df[y_col].astype(int)
    except Exception:
        pass

    # Normalize keys to string
    df[id_col] = df[id_col].astype(str)
    return dict(zip(df[id_col], df[y_col]))

def load_predictions(json_path):
    with open(json_path, "r") as f:
        preds = json.load(f)

    # Expect keys written by our inference: segment_id, pred_idx, confidence
    # (see project script layout)  #
    out = []
    for item in preds:
        seg_id = str(item.get("segment_id"))
        pred   = item.get("pred_idx")  # integer class id
        if pred is None or seg_id is None:
            continue
        try:
            pred = int(pred)
        except Exception:
            pass
        out.append((seg_id, pred))
    return out

def evaluate():
    gt_dict = load_ground_truth(GROUND_TRUTH_FILE)
    preds = load_predictions(PREDICTIONS_FILE)

    y_true, y_pred = [], []
    missing = 0

    for seg_id, pred in preds:
        if seg_id not in gt_dict:
            missing += 1
            # You can print if you want to inspect missing IDs:
            # print(f"[WARN] Ground truth not found for {seg_id}")
            continue
        y_true.append(gt_dict[seg_id])
        y_pred.append(pred)

    if not y_true:
        raise RuntimeError("No overlapping segment IDs between predictions and ground truth.")

    # Ensure numeric (when possible)
    try:
        y_true = [int(x) for x in y_true]
        y_pred = [int(x) for x in y_pred]
    except Exception:
        pass

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Top-1 Accuracy: {acc:.3f}")
    if missing:
        print(f"ℹ️  Skipped {missing} predictions with no ground-truth match.")

    print("\n📊 Classification Report (per-class):")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, max_classes=30, save_path=None):
    from collections import Counter
    # Focus on the most common true classes for readability
    counts = Counter(y_true).most_common(max_classes)
    keep = set([c for c, _ in counts])

    y_true_f = []
    y_pred_f = []
    for yt, yp in zip(y_true, y_pred):
        if yt in keep:
            y_true_f.append(yt)
            y_pred_f.append(yp)

    labels = sorted(set(y_true_f) | set(y_pred_f))
    if not labels:
        print("No labels to plot.")
        return

    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(cm, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Top {max_classes} true classes)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"💾 Saved confusion matrix to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    y_true, y_pred = evaluate()
    # Optional:
    # plot_confusion_matrix(y_true, y_pred, max_classes=30, save_path="results/confusion_matrix.png")
