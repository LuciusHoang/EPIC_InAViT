#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

# ========= Paths (edit if needed) =========
PREDICTIONS_FILE = "results/inavit_predictions.jsonl"   # your JSONL format
GROUND_TRUTH_FILE = "EPIC_100_test_labels.csv"          # new CSV with verb/noun and action_id

# ========= Column candidates (GT) =========
ID_CANDIDATES     = ["uid", "narration_id", "segment_id", "id"]
VERB_CANDIDATES   = ["verb_id", "verb_cls", "verb", "verb_class"]
NOUN_CANDIDATES   = ["noun_id", "noun_cls", "noun", "noun_class"]
ACTION_CANDIDATES = ["action_id", "action_cls", "action"]

# EK-100 NEW ACTION FORMULA
ACTION_BASE = 352   # action_id = verb_id * 352 + noun_id

# --------------------------------------------------------
# I/O helpers
# --------------------------------------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str], required_name: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required_name in ("verb", "noun"):
        # allow missing v/n (we can still do action if given directly)
        return None
    raise KeyError(f"Ground truth CSV must contain a {required_name} column from: {candidates}")

def load_ground_truth(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Loads GT and returns:
      df: dataframe with standardized columns ['sid','verb','noun','action']
      cols: resolved original column names
    """
    df = pd.read_csv(csv_path)

    id_col     = _pick_col(df, ID_CANDIDATES, "segment-id")
    verb_col   = _pick_col(df, VERB_CANDIDATES, "verb")
    noun_col   = _pick_col(df, NOUN_CANDIDATES, "noun")
    action_col = _pick_col(df, ACTION_CANDIDATES, "action label")

    keep = [c for c in [id_col, verb_col, noun_col, action_col] if c]
    df = df[keep].copy()

    # normalize types
    df[id_col] = df[id_col].astype(str)
    if verb_col:   df[verb_col]   = pd.to_numeric(df[verb_col], errors="coerce").astype("Int64")
    if noun_col:   df[noun_col]   = pd.to_numeric(df[noun_col], errors="coerce").astype("Int64")
    if action_col: df[action_col] = pd.to_numeric(df[action_col], errors="coerce").astype("Int64")

    # standardize column names in a copy
    out = pd.DataFrame({"sid": df[id_col].astype(str)})
    out["verb"]   = df[verb_col]   if verb_col   else pd.Series([pd.NA]*len(df), dtype="Int64")
    out["noun"]   = df[noun_col]   if noun_col   else pd.Series([pd.NA]*len(df), dtype="Int64")
    out["action"] = df[action_col] if action_col else pd.Series([pd.NA]*len(df), dtype="Int64")

    return out, {"id": id_col, "verb": verb_col, "noun": noun_col, "action": action_col}

def _iter_json_records(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_predictions(path: str) -> pd.DataFrame:
    """
    Builds a dataframe:
      sid, verb_top5, noun_top5, verb_scores, noun_scores
    """
    rows = []
    for item in _iter_json_records(path):
        sid = None
        for key in ("segment_id", "uid", "narration_id", "id"):
            if key in item and item[key] not in (None, ""):
                sid = str(item[key])
                break
        if sid is None:
            continue

        v5 = item.get("verb_top5_idx", []) or []
        n5 = item.get("noun_top5_idx", []) or []
        vs = item.get("verb_top5_scores", []) or []
        ns = item.get("noun_top5_scores", []) or []

        # cast to ints / floats safely
        try: v5 = [int(x) for x in v5]
        except Exception: pass
        try: n5 = [int(x) for x in n5]
        except Exception: pass
        try: vs = [float(x) for x in vs]
        except Exception: vs = []
        try: ns = [float(x) for x in ns]
        except Exception: ns = []

        rows.append({
            "sid": sid,
            "verb_top5": v5,
            "noun_top5": n5,
            "verb_scores": vs,
            "noun_scores": ns,
        })
    if not rows:
        return pd.DataFrame(columns=["sid","verb_top5","noun_top5","verb_scores","noun_scores"])
    return pd.DataFrame(rows)

# --------------------------------------------------------
# Metrics & helpers
# --------------------------------------------------------
def sample_topk_recall(gt: List[int], pred_topk: List[List[int]], k: int = 5) -> float:
    hits = 0
    n = 0
    for y, tk in zip(gt, pred_topk):
        if tk is None:
            continue
        n += 1
        if y in tk[:k]:
            hits += 1
    return (hits / n) if n else 0.0

def _combine_pairs_topk(v5: List[int], n5: List[int],
                        vs: Optional[List[float]] = None,
                        ns: Optional[List[float]] = None,
                        topk: int = 5) -> List[int]:
    """
    Form all 25 (verb, noun) pairs from top-5 lists.
    Score = (vs[i] * ns[j]) if provided, else 1.0.
    Return top-k action_ids computed as v*352+n.
    """
    cand = []
    for i, v in enumerate(v5[:5]):
        sv = vs[i] if vs and i < len(vs) else 1.0
        for j, n in enumerate(n5[:5]):
            sn = ns[j] if ns and j < len(ns) else 1.0
            a = int(v) * ACTION_BASE + int(n)
            cand.append((a, float(sv * sn)))
    if not cand:
        return []
    cand.sort(key=lambda t: -t[1])
    return [a for (a, _) in cand[:topk]]

# --------------------------------------------------------
# Evaluation (Top-5 verb/noun + derived action)
# --------------------------------------------------------
def evaluate(save_report: bool = True):
    # Load
    gt_df, cols = load_ground_truth(GROUND_TRUTH_FILE)
    pred_df = load_predictions(PREDICTIONS_FILE)

    # Inner join on sid for aligned evaluation
    merged = gt_df.merge(pred_df, on="sid", how="inner")
    print(f"Loaded GT: {len(gt_df)} | Predictions: {len(pred_df)} | Overlap: {len(merged)}")
    print(f"Resolved GT columns: {cols}")

    # ---- Sanity check: does action == verb*352+noun? (when both present) ----
    if merged["verb"].notna().any() and merged["noun"].notna().any() and merged["action"].notna().any():
        calc_action = (merged["verb"].astype("Int64") * ACTION_BASE + merged["noun"].astype("Int64")).astype("Int64")
        mismatch = (merged["action"].notna()) & (calc_action.notna()) & (merged["action"] != calc_action)
        mismatches = int(mismatch.sum())
        if mismatches:
            print(f"⚠️  {mismatches} / {len(merged)} rows where action_id != verb*{ACTION_BASE}+noun in the CSV.")
        else:
            print("✓ CSV action_id matches verb*352+noun for all rows with v/n present.")

    # ---- Verb top-5 ----
    verb_y = []
    verb_p = []
    if merged["verb"].notna().any():
        for _, r in merged.iterrows():
            if pd.isna(r["verb"]):
                continue
            verb_y.append(int(r["verb"]))
            v5 = r["verb_top5"] if isinstance(r["verb_top5"], list) else []
            verb_p.append(v5 if v5 else None)
        verb_t5 = sample_topk_recall(verb_y, verb_p, k=5) if verb_y else 0.0
        print(f"Verb   Top-5 Recall: {verb_t5:.4f}")

    # ---- Noun top-5 ----
    noun_y = []
    noun_p = []
    if merged["noun"].notna().any():
        for _, r in merged.iterrows():
            if pd.isna(r["noun"]):
                continue
            noun_y.append(int(r["noun"]))
            n5 = r["noun_top5"] if isinstance(r["noun_top5"], list) else []
            noun_p.append(n5 if n5 else None)
        noun_t5 = sample_topk_recall(noun_y, noun_p, k=5) if noun_y else 0.0
        print(f"Noun   Top-5 Recall: {noun_t5:.4f}")

    # ---- Action top-5 (derived from verb/noun) ----
    action_y = []
    action_p = []
    detail_rows = []

    # We can only evaluate an action if we have GT verb & noun and prediction top-5 lists
    for _, r in merged.iterrows():
        v_gt = r["verb"]
        n_gt = r["noun"]
        a_gt = r["action"]
        v5   = r["verb_top5"] if isinstance(r["verb_top5"], list) else []
        n5   = r["noun_top5"] if isinstance(r["noun_top5"], list) else []
        vs   = r["verb_scores"] if isinstance(r["verb_scores"], list) else []
        ns   = r["noun_scores"] if isinstance(r["noun_scores"], list) else []

        # Require GT verb & noun to compute the GT action id consistently
        if pd.isna(v_gt) or pd.isna(n_gt):
            detail_rows.append({
                "sid": r["sid"], "reason": "missing_gt_v_or_n",
                "verb_gt": v_gt, "noun_gt": n_gt, "action_gt": a_gt,
                "verb_top5": v5, "noun_top5": n5
            })
            continue

        a_gt_calc = int(v_gt) * ACTION_BASE + int(n_gt)
        action_y.append(a_gt_calc)

        # Build 25 candidates from top-5 lists (if either list is empty, can't eval)
        if not v5 or not n5:
            action_p.append(None)
            detail_rows.append({
                "sid": r["sid"], "reason": "missing_pred_v_or_n",
                "verb_gt": int(v_gt), "noun_gt": int(n_gt), "action_gt_calc": a_gt_calc,
                "verb_top5": v5, "noun_top5": n5
            })
            continue

        a5 = _combine_pairs_topk(v5, n5, vs, ns, topk=5)
        action_p.append(a5)

        hit = (a_gt_calc in (a5[:5] if a5 else []))
        detail_rows.append({
            "sid": r["sid"],
            "verb_gt": int(v_gt), "noun_gt": int(n_gt),
            "action_gt_calc": a_gt_calc,
            "verb_top5": v5, "noun_top5": n5,
            "action_top5_from_pairs": a5,
            "hit": bool(hit)
        })

    action_t5 = sample_topk_recall(action_y, action_p, k=5) if action_y else 0.0
    print(f"Action Top-5 Recall (derived v×n): {action_t5:.4f}  (evaluated on {len(action_y)} samples)")

    # ---- Save report ----
    if save_report:
        os.makedirs("results", exist_ok=True)
        # Summary CSV
        summary = {
            "Top5_Recall_Verb": float(verb_t5) if verb_y else None,
            "Top5_Recall_Noun": float(noun_t5) if noun_y else None,
            "Top5_Recall_Action_Derived": float(action_t5),
            "num_eval_action": int(len(action_y)),
            "overlap": int(len(merged)),
        }
        with open("results/eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Detailed CSV
        pd.DataFrame(detail_rows).to_csv("results/eval_detailed.csv", index=False)

        print("Saved: results/eval_summary.json, results/eval_detailed.csv")

if __name__ == "__main__":
    evaluate(save_report=True)