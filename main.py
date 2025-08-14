#!/usr/bin/env python3
import os
import json
import time
import sys
import argparse
import traceback
from typing import List, Optional, Tuple

import pandas as pd

# Make InAViT importable (expects repo folder "InAViT" in project root)
sys.path.append(os.path.abspath("InAViT"))

from slowfast.config.defaults import get_cfg
from inavit_inference import predict_segment, build_and_load_model


# ========= Defaults (can override via CLI) =========
DEFAULT_CSV = "EPIC_100_test_segments.csv"          # must include id + start/stop cols
DEFAULT_FRAMES_BASE = "/Volumes/T7_Shield/inference/frames_rgb_flow/rgb"
DEFAULT_OBJ_BASE    = "/Volumes/T7_Shield/inference/object_detection_images"
DEFAULT_OUT         = "results/inavit_predictions.jsonl"
DEFAULT_CFG         = "EK_INAVIT_MF_ant.yaml"

# ========= CSV column candidates (flexible to various EK100 lists) =========
ID_CANDIDATES    = ["narration_id", "segment_id", "id", "clip_uid"]
START_CANDIDATES = ["start_frame", "start"]
STOP_CANDIDATES  = ["stop_frame", "stop"]


def _pick_col(df: pd.DataFrame, names: List[str], kind: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"CSV missing {kind} column; expected one of: {names}. "
                   f"Found columns: {list(df.columns)}")


def _load_segments(csv_path: str):
    """
    Load rows with (segment_id, start_frame, stop_frame) from the label CSV.
    Keeps the original row order.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Test list not found: {csv_path}")
    df = pd.read_csv(csv_path)
    id_col    = _pick_col(df, ID_CANDIDATES, "segment id")
    start_col = _pick_col(df, START_CANDIDATES, "start_frame")
    stop_col  = _pick_col(df, STOP_CANDIDATES, "stop_frame")

    use = df[[id_col, start_col, stop_col]].copy()
    use[id_col]    = use[id_col].astype(str)
    use[start_col] = pd.to_numeric(use[start_col], errors="coerce").fillna(0).astype(int)
    use[stop_col]  = pd.to_numeric(use[stop_col], errors="coerce").fillna(0).astype(int)
    return use.to_dict(orient="records"), id_col, start_col, stop_col


def _base_id(segment_id: str) -> str:
    # Map 'P01_11_0' or 'P01_11' → 'P01_11'
    return "_".join(segment_id.split("_")[:2])


def _frames_dir_for(segment_id: str, frames_base: str) -> str:
    base_id = _base_id(segment_id)
    participant = base_id.split("_")[0]
    return os.path.join(frames_base, participant, base_id)


def _obj_dir_for(segment_id: str, obj_base: str) -> str:
    base_id = _base_id(segment_id)
    participant = base_id.split("_")[0]
    return os.path.join(obj_base, participant, base_id)


def _write_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _parse_id_filter(ids_arg: Optional[str]) -> Optional[set]:
    """
    Allow passing a comma-separated list of ids OR a text file path with one id per line.
    """
    if not ids_arg:
        return None
    if os.path.isfile(ids_arg):
        with open(ids_arg, "r") as f:
            return {line.strip() for line in f if line.strip()}
    # Treat as comma-separated list
    return {x.strip() for x in ids_arg.split(",") if x.strip()}


def _safe_top1(lst, default_val=None):
    """Return lst[0] if possible, else a default (None or float('nan'))."""
    try:
        return lst[0]
    except Exception:
        return default_val


def _sanitize_topk_dict(out: dict, seg_id: str, anticipation: bool) -> dict:
    """
    Ensure the predict_segment() output has the required keys and non-empty lists.
    In anticipation mode, we only enforce action; verb/noun are optional and
    default to empty lists without warnings.
    """
    fixed = dict(out)  # shallow copy

    # Always ensure action exists
    for k_idx, k_scr in [("action_top5_idx", "action_top5_scores")]:
        if k_idx not in fixed or not isinstance(fixed[k_idx], list) or len(fixed[k_idx]) == 0:
            print(f"[WARN] {seg_id}: Empty top‑k for action. Filling with sentinel.")
            fixed[k_idx] = [-1]
            fixed[k_scr] = [0.0]
        elif k_scr not in fixed or not isinstance(fixed[k_scr], list) or len(fixed[k_scr]) == 0:
            print(f"[WARN] {seg_id}: Missing action scores. Filling with sentinel.")
            fixed[k_idx] = [-1]
            fixed[k_scr] = [0.0]

    # Verb / noun handling
    if anticipation:
        # In anticipation, verb/noun may be absent; just normalize to empty lists.
        fixed.setdefault("verb_top5_idx", [])
        fixed.setdefault("verb_top5_scores", [])
        fixed.setdefault("noun_top5_idx", [])
        fixed.setdefault("noun_top5_scores", [])
    else:
        # For recognition runs, enforce presence; if empty, fill sentinel.
        for k_idx, k_scr in [("verb_top5_idx","verb_top5_scores"), ("noun_top5_idx","noun_top5_scores")]:
            if k_idx not in fixed or not isinstance(fixed[k_idx], list) or len(fixed[k_idx]) == 0:
                print(f"[WARN] {seg_id}: Empty top‑k for {k_idx.replace('_top5_idx','')}. Filling with sentinel.")
                fixed[k_idx] = [-1]
                fixed[k_scr] = [0.0]
            elif k_scr not in fixed or not isinstance(fixed[k_scr], list) or len(fixed[k_scr]) == 0:
                print(f"[WARN] {seg_id}: Missing scores for {k_idx.replace('_top5_idx','')}. Filling with sentinel.")
                fixed[k_idx] = [-1]
                fixed[k_scr] = [0.0]

    return fixed


def build_cfg(cfg_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    return cfg


def main():
    parser = argparse.ArgumentParser(description="InAViT | EK100 Anticipation Inference")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to EPIC test segments CSV")
    parser.add_argument("--frames-base", default=DEFAULT_FRAMES_BASE, help="RGB frames root")
    parser.add_argument("--obj-base", default=DEFAULT_OBJ_BASE, help="Object-detection frames root (optional)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--cfg", default=DEFAULT_CFG, help="Model/config YAML")
    parser.add_argument("--weights", default=None, help="Optional checkpoint path override")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N segments")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N segments")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated ids OR path to file containing one id per line to run only a subset")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing JSONL instead of overwriting it")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    # macOS/CPU stability
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # ---- Build config & model ----
    cfg = build_cfg(args.cfg)
    # Determine anticipation mode (used to silence verb/noun in logs)
    is_ant = bool(getattr(cfg, "EPICKITCHENS", None) and getattr(cfg.EPICKITCHENS, "ANTICIPATION", False))

    if args.weights:
        model = build_and_load_model(cfg, pretrained_path=args.weights)
    else:
        model = build_and_load_model(cfg)
    model.eval()

    # ---- Load segments ----
    rows, id_col, start_col, stop_col = _load_segments(args.csv)

    # Optional ID filter
    include_ids = _parse_id_filter(args.ids)
    if include_ids is not None:
        rows = [r for r in rows if str(r[id_col]) in include_ids]

    # Skip / limit
    if args.skip:
        rows = rows[args.skip:]
    if args.limit is not None:
        rows = rows[: args.limit]

    total = len(rows)
    if total == 0:
        print("[WARN] No rows to process after filters. Exiting.")
        return

    # Prepare output file
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not args.append:
        open(args.out, "w").close()

    t0 = time.time()
    processed = 0
    missing_rgb = 0
    failures = 0

    if not args.quiet:
        print(f"[INFO] Subset size: {total}")
        print(f"[INFO] CSV: {args.csv}")
        print(f"[INFO] Frames base: {args.frames_base}")
        print(f"[INFO] Obj base: {args.obj_base}")
        print(f"[INFO] Writing JSONL to: {args.out}")
        print(f"[INFO] Anticipation mode: {is_ant}")

    for i, row in enumerate(rows, 1):
        segment_id  = row[id_col]
        start_frame = int(row[start_col])
        stop_frame  = int(row[stop_col])

        frames_dir = _frames_dir_for(segment_id, args.frames_base)
        if not os.path.isdir(frames_dir):
            print(f"[WARN] Missing RGB frames dir → {frames_dir}")
            missing_rgb += 1
            continue

        # object detections are optional; if not present, predict_segment will create dummy boxes
        obj_dir = _obj_dir_for(segment_id, args.obj_base)
        if not os.path.isdir(obj_dir):
            obj_dir = None

        if not args.quiet:
            print(f"\n▶️ [{i}/{total}] {segment_id}")

        try:
            # ---------- Inference (protocol enforced inside predict_segment) ----------
            out_raw = predict_segment(
                frames_dir=frames_dir,
                obj_dir=obj_dir,
                model=model,
                cfg=cfg,
                start_frame=start_frame,
                stop_frame=stop_frame,
            )

            # Normalize and fill sentinels only where appropriate
            out = _sanitize_topk_dict(out_raw, segment_id, anticipation=is_ant)

            rec = {
                "clip_uid": segment_id,
                "video_id": _base_id(segment_id),
                "verb_top5":   out["verb_top5_idx"],
                "verb_scores": [float(x) for x in out["verb_top5_scores"]],
                "noun_top5":   out["noun_top5_idx"],
                "noun_scores": [float(x) for x in out["noun_top5_scores"]],
                "action_top5": out["action_top5_idx"],
                "action_scores": [float(x) for x in out["action_top5_scores"]],
                # quick-look single-line summary
                "pred1": {
                    "head": "action",
                    "class": int(_safe_top1(out["action_top5_idx"], -1) if out["action_top5_idx"] else -1),
                    "conf":  float(_safe_top1(out["action_top5_scores"], 0.0) if out["action_top5_scores"] else 0.0),
                },
            }
            _write_jsonl(args.out, rec)

            processed += 1
            elapsed = time.time() - t0
            avg = elapsed / max(processed, 1)
            eta = avg * (total - i)

            # Progress logging
            act1  = rec["action_top5"][0]  if rec["action_top5"]  else -1
            act1s = rec["action_scores"][0] if rec["action_scores"] else 0.0

            if is_ant:
                # Anticipation: action only
                if not args.quiet:
                    print(f"act@1={act1}({act1s:.4f})")
            else:
                verb1  = rec["verb_top5"][0]  if rec["verb_top5"]  else -1
                verb1s = rec["verb_scores"][0] if rec["verb_scores"] else 0.0
                noun1  = rec["noun_top5"][0]  if rec["noun_top5"]  else -1
                noun1s = rec["noun_scores"][0] if rec["noun_scores"] else 0.0
                if not args.quiet:
                    print("act@1={}({:.4f}) | verb@1={}({:.3f}) | noun@1={}({:.3f})"
                          .format(act1, act1s, verb1, verb1s, noun1, noun1s))

            if not args.quiet:
                print(f"⏱  elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

        except Exception as e:
            failures += 1
            print(f"[ERROR] Inference failed for {segment_id}: {e}")
            if not args.quiet:
                tb = "".join(traceback.format_exception_only(type(e), e)).strip()
                print(f"[TRACE] {tb}")
            # Optionally log an error record for alignment
            fail_rec = {
                "clip_uid": segment_id,
                "video_id": _base_id(segment_id),
                "error": str(e),
            }
            _write_jsonl(args.out, fail_rec)

    print("\n✅ Inference complete →", args.out)
    if missing_rgb:
        print(f"ℹ️ Skipped {missing_rgb} segments due to missing RGB frame folders.")
    if failures:
        print(f"⚠️  {failures} segments encountered errors (wrote error records).")
    print(f"Processed successfully: {processed}/{total}")


if __name__ == "__main__":
    main()