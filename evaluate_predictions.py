import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


# ========= Paths (edit if needed) =========
PREDICTIONS_FILE = "results/inavit_predictions.json"

# Ground truth CSV can be as small as:
#   - uid, action_id
# or richer versions with verb/noun columns.
GROUND_TRUTH_FILE = "EPIC_100_test_labels.csv"

# ========= Column candidates =========
# (Updated: prefer 'uid' first; also accept narration_id/segment_id/id)
ID_CANDIDATES = ["uid", "narration_id", "segment_id", "id"]
VERB_CANDIDATES = ["verb_id", "verb_cls", "verb", "verb_class"]
NOUN_CANDIDATES = ["noun_id", "noun_cls", "noun", "noun_class"]
ACTION_CANDIDATES = ["action_id", "action_cls", "action"]


# -----------------------------
# Helpers
# -----------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str], kind: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # It's okay to return None for verb/noun if your CSV doesn't have them
    if kind in ("verb", "noun"):
        return None
    raise KeyError(f"Ground truth CSV must contain a {kind} column in {candidates}")


def load_ground_truth(csv_path: str) -> Tuple[Dict[str, Dict[str, Optional[int]]], Dict[str, str]]:
    """
    Returns:
      gt: {segment_id_str: {'verb': int|None, 'noun': int|None, 'action': int}}
      cols: resolved column names
    """
    df = pd.read_csv(csv_path)

    id_col = _pick_col(df, ID_CANDIDATES, "segment-id")
    verb_col = _pick_col(df, VERB_CANDIDATES, "verb")
    noun_col = _pick_col(df, NOUN_CANDIDATES, "noun")
    action_col = _pick_col(df, ACTION_CANDIDATES, "action label")

    keep_cols = [c for c in [id_col, verb_col, noun_col, action_col] if c is not None]
    df = df[keep_cols].dropna(subset=[id_col, action_col])

    # enforce types
    df[id_col] = df[id_col].astype(str)
    if verb_col:   df[verb_col] = df[verb_col].astype(int)
    if noun_col:   df[noun_col] = df[noun_col].astype(int)
    df[action_col] = df[action_col].astype(int)

    gt = {}
    for _, r in df.iterrows():
        sid = str(r[id_col])
        gt[sid] = {
            "verb": int(r[verb_col]) if verb_col else None,
            "noun": int(r[noun_col]) if noun_col else None,
            "action": int(r[action_col]),
        }
    cols = {"id": id_col, "verb": verb_col, "noun": noun_col, "action": action_col}
    return gt, cols


def load_predictions(json_path: str):
    """
    Accepts two formats:

    (A) Action-only:
      {"segment_id": "P01_11_000", "pred_idx": 1234, "confidence": 0.77}
      or with id under "uid"/"narration_id"/"id"

    (B) Full InAViT-style:
      {
        "segment_id": "...",   # or uid/narration_id/id
        "verb_top5":   [v1,...,v5],   "verb_scores": [s1,...,s5],
        "noun_top5":   [n1,...,n5],   "noun_scores": [s1,...,s5],
        "action_top5": [a1,...,a5],   "action_scores":[s1,...,s5]
      }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    preds = {}
    for item in raw:
        # robust id key detection
        sid = None
        for k in ("segment_id", "uid", "narration_id", "id"):
            if k in item and item[k] not in (None, ""):
                sid = str(item[k])
                break
        if not sid:
            continue

        record = {}

        # Full heads (preferred)
        for head in ("verb", "noun", "action"):
            tk = f"{head}_top5"
            if tk in item and isinstance(item[tk], list) and len(item[tk]) > 0:
                # coerce to ints where possible
                top5 = []
                for x in item[tk]:
                    try:
                        top5.append(int(x))
                    except Exception:
                        top5.append(x)
                record[head] = {"top5": top5}

        # Action-only fallback (top1)
        if "action" not in record:
            # accept either "pred_idx" or "action_id" for single-head predictions
            for key in ("pred_idx", "action_id"):
                if key in item:
                    try:
                        record["action"] = {"top1": int(item[key])}
                    except Exception:
                        record["action"] = {"top1": item[key]}
                    break

        preds[sid] = record

    return preds


def topk_recall(gt: List[int], pred_topk: List[List[int]], k: int = 5) -> float:
    hit = 0
    for y, topk in zip(gt, pred_topk):
        if y in (topk[:k] if topk is not None else []):
            hit += 1
    return hit / max(1, len(gt))


def mean_topk_recall(gt: List[int], pred_topk: List[List[int]], k: int = 5) -> float:
    """
    Class-averaged top-k recall over the classes present in gt.
    """
    per_class_hits = Counter()
    per_class_counts = Counter()

    for y, topk in zip(gt, pred_topk):
        per_class_counts[y] += 1
        if y in (topk[:k] if topk is not None else []):
            per_class_hits[y] += 1

    recalls = []
    for c in per_class_counts:
        recalls.append(per_class_hits[c] / per_class_counts[c])
    return float(np.mean(recalls)) if recalls else 0.0


def _gather_pairs(
    gt_map: Dict[str, Dict[str, Optional[int]]],
    preds_map: Dict[str, dict],
    head: str,
) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Returns:
      y_true_top1, y_pred_top1, y_pred_top5_lists
    Some entries may miss top1 or top5 depending on your prediction format.
    """
    y_true, y_pred1, y_pred5 = [], [], []
    for sid, gt_rec in gt_map.items():
        if sid not in preds_map:
            continue
        pred_rec = preds_map[sid].get(head, None)
        y = gt_rec.get(head, None)
        if y is None or pred_rec is None:
            continue

        # true label
        try:
            y = int(y)
        except Exception:
            pass

        # predicted
        top1 = None
        top5 = None
        if "top5" in pred_rec and pred_rec["top5"]:
            top5 = pred_rec["top5"]
            top1 = top5[0]
        if "top1" in pred_rec and pred_rec["top1"] is not None:
            top1 = pred_rec["top1"]

        if top1 is not None:
            try:
                top1 = int(top1)
            except Exception:
                pass

        # Append
        y_true.append(y)
        y_pred1.append(top1 if top1 is not None else -1)
        y_pred5.append(top5 if top5 is not None else [])
    return y_true, y_pred1, y_pred5


def _print_headline(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# -----------------------------
# Public API
# -----------------------------
def evaluate(save_report: bool = True):
    # Load
    gt_map, cols = load_ground_truth(GROUND_TRUTH_FILE)
    preds_map = load_predictions(PREDICTIONS_FILE)

    total_pred = len(preds_map)
    overlap = len(set(gt_map.keys()) & set(preds_map.keys()))
    print(f"Loaded GT: {len(gt_map)} segments | Predictions: {total_pred} | Overlap: {overlap}")
    print(f"Resolved GT columns: {cols}")

    report_rows = []
    summary = {}

    for head, nice in [("action", "Action"), ("verb", "Verb"), ("noun", "Noun")]:
        y_true, y_pred1, y_pred5 = _gather_pairs(gt_map, preds_map, head)

        if not y_true:
            continue

        _print_headline(f"{nice} metrics")

        # Top-1 (sample-avg)
        y_true_t1, y_pred_t1 = [], []
        for t, p in zip(y_true, y_pred1):
            if p is not None and p != -1:
                y_true_t1.append(t)
                y_pred_t1.append(p)
        top1 = accuracy_score(y_true_t1, y_pred_t1) if y_true_t1 else 0.0
        print(f"Top-1 accuracy (sample-avg): {top1:.3f}")

        # Top-5 recall (sample-avg) + Mean Top-5 (class-avg)
        has_top5 = any(len(lst) > 0 for lst in y_pred5)
        if has_top5:
            t5 = topk_recall(y_true, y_pred5, k=5)
            mt5 = mean_topk_recall(y_true, y_pred5, k=5)
            print(f"Top-5 recall  (sample-avg): {t5:.3f}")
            print(f"Mean Top-5 recall (class-avg over classes present in subset): {mt5:.3f}")
        else:
            t5, mt5 = None, None
            print("Top-5 lists not found in predictions → skipping Top-5 metrics for this head.")

        # Brief per-class report (Top-1)
        try:
            print("\nPer-class precision/recall/f1 (Top-1) — truncated to first 10 classes:")
            cr = classification_report(y_true_t1, y_pred_t1, digits=3, zero_division=0, output_dict=True)
            printed = 0
            for cls_id, stats in cr.items():
                if isinstance(stats, dict) and cls_id not in ("accuracy", "macro avg", "weighted avg"):
                    print(f"  class {cls_id:>4}: p={stats.get('precision',0):.3f} "
                          f"r={stats.get('recall',0):.3f} f1={stats.get('f1-score',0):.3f} "
                          f"n={stats.get('support',0)}")
                    printed += 1
                    if printed >= 10:
                        break
        except Exception:
            pass

        # Collect summary & row
        summary[f"{nice}_Top1"] = float(top1)
        if has_top5:
            summary[f"{nice}_Top5"] = float(t5)
            summary[f"{nice}_MeanTop5"] = float(mt5)

        report_rows.append({
            "head": nice.lower(),
            "num_pairs": len(y_true),
            "top1_acc": float(top1),
            "top5_recall": float(t5) if t5 is not None else np.nan,
            "mean_top5_recall": float(mt5) if mt5 is not None else np.nan,
        })

    # Save a compact CSV report if requested
    if save_report and report_rows:
        rep_df = pd.DataFrame(report_rows)
        rep_df.to_csv("results/eval_report.csv", index=False)
        with open("results/eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nSaved: results/eval_report.csv and results/eval_summary.json")


if __name__ == "__main__":
    evaluate(save_report=True)