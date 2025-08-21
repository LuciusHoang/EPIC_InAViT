import os
import json
from typing import Dict, List, Tuple, Optional, Any, Iterable

import numpy as np
import pandas as pd

# ========= Paths (edit if needed) =========
PREDICTIONS_FILE = "results/inavit_predictions.json"   # .json OR .jsonl supported
GROUND_TRUTH_FILE = "EPIC_100_test_labels.csv"

# ========= Column candidates (GT) =========
ID_CANDIDATES = ["uid", "narration_id", "segment_id", "id"]
VERB_CANDIDATES = ["verb_id", "verb_cls", "verb", "verb_class"]
NOUN_CANDIDATES = ["noun_id", "noun_cls", "noun", "noun_class"]
ACTION_CANDIDATES = ["action_id", "action_cls", "action"]

# --------------------------------------------------------
# I/O helpers
# --------------------------------------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str], required_name: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required_name in ("verb", "noun"):
        return None
    raise KeyError(f"Ground truth CSV must contain a {required_name} column from: {candidates}")

def load_ground_truth(csv_path: str) -> Tuple[Dict[str, Dict[str, Optional[int]]], Dict[str, str]]:
    """
    Returns:
      gt_map: {segment_id_str: {'verb': int|None, 'noun': int|None, 'action': int}}
      cols: resolved column names
    """
    df = pd.read_csv(csv_path)

    id_col = _pick_col(df, ID_CANDIDATES, "segment-id")
    verb_col = _pick_col(df, VERB_CANDIDATES, "verb")
    noun_col = _pick_col(df, NOUN_CANDIDATES, "noun")
    action_col = _pick_col(df, ACTION_CANDIDATES, "action label")

    keep = [c for c in [id_col, verb_col, noun_col, action_col] if c]
    df = df[keep].dropna(subset=[id_col, action_col])

    df[id_col] = df[id_col].astype(str)
    if verb_col: df[verb_col] = df[verb_col].astype(int)
    if noun_col: df[noun_col] = df[noun_col].astype(int)
    df[action_col] = df[action_col].astype(int)

    gt_map: Dict[str, Dict[str, Optional[int]]] = {}
    for _, r in df.iterrows():
        sid = str(r[id_col])
        gt_map[sid] = {
            "verb": int(r[verb_col]) if verb_col else None,
            "noun": int(r[noun_col]) if noun_col else None,
            "action": int(r[action_col]),
        }
    return gt_map, {"id": id_col, "verb": verb_col, "noun": noun_col, "action": action_col}

def _iter_json_records(path: str) -> Iterable[dict]:
    """
    Supports:
      - JSON list  (file ends with .json)
      - JSON Lines (file ends with .jsonl OR content is newline-delimited objects)
    """
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    # Try as a single JSON doc
    with open(path, "r") as f:
        txt = f.read().strip()
    if not txt:
        return
    # If it looks like a list, load once
    if txt[0] == "[":
        data = json.loads(txt)
        for rec in data:
            yield rec
        return
    # Otherwise assume jsonl but misnamed
    for line in txt.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)

def _first_nonempty(d: dict, keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None

def load_predictions(path: str) -> Dict[str, dict]:
    """
    Builds a map: preds[sid] = {
        'verb':   {'top5': [..], 'scores': [..]} OR {'logits': [..]},
        'noun':   {...},
        'action': {...}   (top5/scores or logits or top1 fallback)
    }
    Accepts keys:
      - top5 arrays:    '*_top5_idx', '*_top5'
      - scores arrays:  '*_top5_scores', '*_scores'
      - raw logits:     '*_logits'
      - ids:            'segment_id', 'uid', 'narration_id', 'id'
      - action top1:    'pred_idx' or 'action_id' (fallback)
    """
    preds: Dict[str, dict] = {}
    for item in _iter_json_records(path):
        sid = _first_nonempty(item, ["segment_id", "uid", "narration_id", "id"])
        if sid is None:
            continue
        sid = str(sid)

        rec = preds.get(sid, {})

        for head in ("verb", "noun", "action"):
            h = rec.get(head, {})

            # logits (preferred for deriving top-5 reliably)
            logits = item.get(f"{head}_logits")
            if isinstance(logits, list) and logits:
                h["logits"] = logits

            # top5 indices (various key names)
            t5 = _first_nonempty(item, [f"{head}_top5_idx", f"{head}_top5"])
            if isinstance(t5, list) and t5:
                try:
                    t5 = [int(x) for x in t5]
                except Exception:
                    pass
                h["top5"] = t5

            # top5 scores (optional)
            sc = _first_nonempty(item, [f"{head}_top5_scores", f"{head}_scores"])
            if isinstance(sc, list) and sc and (h.get("top5") and len(sc) == len(h["top5"])):
                h["scores"] = sc

            # action top1 fallback
            if head == "action" and "top5" not in h and "logits" not in h:
                a1 = _first_nonempty(item, ["pred_idx", "action_id"])
                if a1 is not None:
                    try:
                        a1 = int(a1)
                    except Exception:
                        pass
                    h["top1"] = a1

            if h:
                rec[head] = h

        preds[sid] = rec
    return preds

# --------------------------------------------------------
# Metrics
# --------------------------------------------------------
def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x, dtype=np.float64)
    return e / np.sum(e)

def _topk_from_logits(logits: List[float], k: int) -> List[int]:
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    # argpartition for speed, then sort the selected
    if k >= arr.size:
        order = np.argsort(-arr)
        return order.tolist()
    idx = np.argpartition(-arr, kth=k-1)[:k]
    # sort those by value desc
    idx = idx[np.argsort(-arr[idx])]
    return idx.tolist()

def sample_topk_recall(gt: List[int], pred_topk: List[List[int]], k: int = 5) -> float:
    """Sample-averaged Top-k recall."""
    hits = 0
    n = 0
    for y, tk in zip(gt, pred_topk):
        if tk is None:
            continue
        n += 1
        if y in tk[:k]:
            hits += 1
    return (hits / n) if n else 0.0

# --------------------------------------------------------
# Joint action top-5 from verb/noun
# --------------------------------------------------------
def joint_action_top5_from_logits(verb_logits: List[float], noun_logits: List[float],
                                  kv: int = 20, kn: int = 20) -> List[Tuple[int, int]]:
    """Return top-5 (verb_id, noun_id) by product of softmax probabilities."""
    vp = _softmax_1d(np.asarray(verb_logits, dtype=np.float64))
    np_ = _softmax_1d(np.asarray(noun_logits, dtype=np.float64))

    # prune to Kv, Kn for speed
    kv = min(kv, vp.size)
    kn = min(kn, np_.size)
    topv = _topk_from_logits(vp.tolist(), kv)
    topn = _topk_from_logits(np_.tolist(), kn)

    cand: List[Tuple[Tuple[int, int], float]] = []
    for v in topv:
        pv = vp[v]
        # multiply across all kept nouns
        scores = pv * np_[topn]  # vectorized
        for j, n in enumerate(topn):
            cand.append(((v, n), float(scores[j])))

    cand.sort(key=lambda x: -x[1])
    return [vn for (vn, _) in cand[:5]]

def joint_action_top5_from_lists(verb_top5: List[int],
                                 noun_top5: List[int],
                                 verb_scores: Optional[List[float]] = None,
                                 noun_scores: Optional[List[float]] = None) -> List[Tuple[int, int]]:
    """
    If we only have top-5 lists (and optionally scores), form all 25 pairs and take the best 5.
    If scores are missing, treat each verb/noun top-5 as equal weight.
    """
    cand: List[Tuple[Tuple[int, int], float]] = []
    for i, v in enumerate(verb_top5):
        for j, n in enumerate(noun_top5):
            sv = verb_scores[i] if (verb_scores and i < len(verb_scores)) else 1.0
            sn = noun_scores[j] if (noun_scores and j < len(noun_scores)) else 1.0
            cand.append(((v, n), float(sv * sn)))
    cand.sort(key=lambda x: -x[1])
    return [vn for (vn, _) in cand[:5]]

# --------------------------------------------------------
# Pairing GT & Preds
# --------------------------------------------------------
def _gather_head_top5(gt_map: Dict[str, Dict[str, Optional[int]]],
                      preds_map: Dict[str, dict],
                      head: str) -> Tuple[List[int], List[Optional[List[int]]]]:
    """
    Returns:
      y_true: list[int]
      pred_top5: list[list[int] or None] aligned to y_true
    """
    y_true: List[int] = []
    pred_t5: List[Optional[List[int]]] = []
    for sid, gt in gt_map.items():
        if head not in gt or gt[head] is None:
            continue
        y = int(gt[head])

        pred = preds_map.get(sid, {}).get(head, {})
        t5 = None

        # direct top-5
        if isinstance(pred.get("top5"), list) and pred["top5"]:
            t5 = [int(x) if isinstance(x, (int, np.integer, str)) and str(x).isdigit() else x
                  for x in pred["top5"]]

        # derive from logits if needed
        elif isinstance(pred.get("logits"), list) and pred["logits"]:
            t5 = _topk_from_logits(pred["logits"], 5)

        y_true.append(y)
        pred_t5.append(t5)
    return y_true, pred_t5

def _gather_action_top5(gt_map: Dict[str, Dict[str, Optional[int]]],
                        preds_map: Dict[str, dict]) -> Tuple[List[Tuple[Optional[int], Optional[int], int]],
                                                             List[Optional[List[int]]],
                                                             List[Optional[List[Tuple[int, int]]]]]:
    """
    Returns:
      gt_triplets: [(verb_gt or None, noun_gt or None, action_gt)]
      action_pred_top5_ids: [list[int] or None]  (when model gave flat action top-5)
      action_pred_top5_pairs: [list[(v,n)] or None] (when derived jointly)
    """
    gt_triplets: List[Tuple[Optional[int], Optional[int], int]] = []
    act_top5_ids: List[Optional[List[int]]] = []
    act_top5_pairs: List[Optional[List[Tuple[int, int]]]] = []

    for sid, gt in gt_map.items():
        a_gt = gt.get("action", None)
        if a_gt is None:
            continue

        pred = preds_map.get(sid, {}).get("action", {})
        # 1) direct action top-5 (if provided)
        t5_ids = None
        if isinstance(pred.get("top5"), list) and pred["top5"]:
            t5_ids = [int(x) if isinstance(x, (int, np.integer, str)) and str(x).isdigit() else x
                      for x in pred["top5"]]

        # 2) joint from logits or from top-5 lists if not provided
        t5_pairs = None
        if t5_ids is None:
            vpred = preds_map.get(sid, {}).get("verb", {})
            npred = preds_map.get(sid, {}).get("noun", {})

            # Prefer logits (accurate)
            if isinstance(vpred.get("logits"), list) and isinstance(npred.get("logits"), list):
                t5_pairs = joint_action_top5_from_logits(vpred["logits"], npred["logits"])
            # Fall back to lists (+optional scores)
            elif isinstance(vpred.get("top5"), list) and isinstance(npred.get("top5"), list):
                t5_pairs = joint_action_top5_from_lists(
                    vpred["top5"], npred["top5"], vpred.get("scores"), npred.get("scores")
                )

        gt_triplets.append((gt.get("verb"), gt.get("noun"), a_gt))
        act_top5_ids.append(t5_ids)
        act_top5_pairs.append(t5_pairs)

    return gt_triplets, act_top5_ids, act_top5_pairs

# --------------------------------------------------------
# Evaluation (Top-5 only)
# --------------------------------------------------------
def evaluate_top5_only(save_report: bool = True):
    # Load
    gt_map, cols = load_ground_truth(GROUND_TRUTH_FILE)
    preds_map = load_predictions(PREDICTIONS_FILE)

    overlap = len(set(gt_map.keys()) & set(preds_map.keys()))
    print(f"Loaded GT: {len(gt_map)} | Predictions: {len(preds_map)} | Overlap: {overlap}")
    print(f"Resolved GT columns: {cols}")

    summary: Dict[str, float] = {}
    rows: List[dict] = []

    # ---- Verb ----
    yv_true, yv_pred5 = _gather_head_top5(gt_map, preds_map, "verb")
    verb_t5 = sample_topk_recall(yv_true, yv_pred5, k=5) if yv_true else 0.0
    if yv_true:
        print(f"Verb   Top-5 Recall: {verb_t5:.4f}")
        summary["Top5_Recall_Verb"] = float(verb_t5)
        rows.append({"head": "verb", "num_samples": len(yv_true), "top5_recall": float(verb_t5)})

    # ---- Noun ----
    yn_true, yn_pred5 = _gather_head_top5(gt_map, preds_map, "noun")
    noun_t5 = sample_topk_recall(yn_true, yn_pred5, k=5) if yn_true else 0.0
    if yn_true:
        print(f"Noun   Top-5 Recall: {noun_t5:.4f}")
        summary["Top5_Recall_Noun"] = float(noun_t5)
        rows.append({"head": "noun", "num_samples": len(yn_true), "top5_recall": float(noun_t5)})

    # ---- Action ----
    gt_triplets, act_top5_ids, act_top5_pairs = _gather_action_top5(gt_map, preds_map)

    # Strategy:
    #  A) If action top-5 IDs are present, use those against action_gt.
    #  B) Else if (verb_gt, noun_gt) exist and we have joint top-5 pairs, treat hit if pair is present.
    action_hits = 0
    action_count = 0

    for (v_gt, n_gt, a_gt), t5_ids, t5_pairs in zip(gt_triplets, act_top5_ids, act_top5_pairs):
        if t5_ids is not None:
            # direct flat action top-5
            action_count += 1
            if a_gt in t5_ids[:5]:
                action_hits += 1
        elif t5_pairs is not None and v_gt is not None and n_gt is not None:
            action_count += 1
            if (int(v_gt), int(n_gt)) in t5_pairs[:5]:
                action_hits += 1
        else:
            # cannot evaluate this sample for action top-5
            continue

    action_t5 = (action_hits / action_count) if action_count else 0.0
    if action_count:
        print(f"Action Top-5 Recall: {action_t5:.4f}  (evaluated on {action_count} samples)")
        summary["Top5_Recall_Action"] = float(action_t5)
        rows.append({"head": "action", "num_samples": int(action_count), "top5_recall": float(action_t5)})

    # ---- Save report ----
    if save_report:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv("results/eval_report.csv", index=False)
        with open("results/eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved: results/eval_report.csv, results/eval_summary.json")

if __name__ == "__main__":
    evaluate_top5_only(save_report=True)