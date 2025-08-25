#!/usr/bin/env python3
import os
import json
import time
import sys
import argparse
import traceback
from typing import List, Optional, Dict, Any

import pandas as pd
import torch

# Make InAViT importable (expects repo folder "InAViT" in project root)
sys.path.append(os.path.abspath("InAViT"))

from slowfast.config.defaults import get_cfg
from inavit_inference import predict_segment, build_and_load_model

# ========= Defaults (override via CLI) =========
DEFAULT_CSV = "EPIC_100_test_segments.csv"          # must include id + start/stop cols
DEFAULT_FRAMES_BASE = "/Volumes/T7_Shield/inference/frames_rgb_flow/rgb"
DEFAULT_OBJ_BASE    = "/Volumes/T7_Shield/inference/object_detection_images"
DEFAULT_OUT         = "results/inavit_predictions.jsonl"
DEFAULT_CFG         = "EK_INAVIT_MF_ant.yaml"

# ========= CSV column candidates (flexible to various EK100 lists) =========
ID_CANDIDATES    = ["narration_id", "segment_id", "id", "clip_uid", "uid"]
START_CANDIDATES = ["start_frame", "start"]
STOP_CANDIDATES  = ["stop_frame", "stop"]


# -------------------------------------------------
# CSV helpers
# -------------------------------------------------
def _pick_col(df: pd.DataFrame, names: List[str], kind: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(
        f"CSV missing {kind} column; expected one of: {names}. "
        f"Found columns: {list(df.columns)}"
    )


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


# -------------------------------------------------
# Path helpers (EK100 layout on your disk)
# -------------------------------------------------
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


# -------------------------------------------------
# Misc
# -------------------------------------------------
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
    """Return lst[0] if possible, else a default."""
    try:
        return lst[0]
    except Exception:
        return default_val


def _write_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# -------------------------------------------------
# Config
# -------------------------------------------------
def build_cfg(cfg_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    return cfg


# -------------------------------------------------
# Model head probing (verb/noun only)
# -------------------------------------------------
def _heads_present(model) -> Dict[str, bool]:
    """
    Return which classifier heads exist on the model (best-effort). Only verb/noun here.
    """
    present = {"verb": False, "noun": False}
    name_map = {
        "verb": ["head0", "verb_head", "head_verb"],
        "noun": ["head1", "noun_head", "head_noun"],
    }
    for key, names in name_map.items():
        for n in names:
            mod = getattr(model, n, None)
            if isinstance(mod, torch.nn.Linear):
                present[key] = True
                break
    return present


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InAViT | EK100 Anticipation Inference (Top-5, verb/noun only)")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to EPIC test segments CSV")
    parser.add_argument("--frames-base", default=DEFAULT_FRAMES_BASE, help="RGB frames root")
    parser.add_argument("--obj-base", default=DEFAULT_OBJ_BASE, help="Object-detection frames root (optional)")
    parser.add_argument("--bboxes", default=None,
                        help="Path to a global bboxes.json (flat or nested). If omitted, falls back to per-segment obj_dir/bboxes.json.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--cfg", default=DEFAULT_CFG, help="Model/config YAML")
    parser.add_argument("--weights", default=None, help="Optional checkpoint path override")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N segments")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N segments")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated ids OR path to file with one id per line to run a subset")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing JSONL instead of overwriting it")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    # macOS/CPU stability
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # ---- Build config & model ----
    cfg = build_cfg(args.cfg)
    is_ant = bool(getattr(cfg, "EPICKITCHENS", None) and getattr(cfg.EPICKITCHENS, "ANTICIPATION", False))
    model = build_and_load_model(cfg, pretrained_path=args.weights) if args.weights else build_and_load_model(cfg)
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not args.append:
        open(args.out, "w").close()

    # Validate a global bboxes path if provided
    global_bboxes = None
    if args.bboxes:
        if os.path.isfile(args.bboxes):
            global_bboxes = os.path.abspath(args.bboxes)
        else:
            print(f"[WARN] --bboxes provided but not found on disk: {args.bboxes} (will fallback per segment)")

    t0 = time.time()
    processed = 0
    missing_rgb = 0
    failures = 0

    if not args.quiet:
        print(f"[INFO] Subset size: {total}")
        print(f"[INFO] CSV: {args.csv}")
        print(f"[INFO] Frames base: {args.frames_base}")
        print(f"[INFO] Obj base: {args.obj_base}")
        print(f"[INFO] Global bboxes: {global_bboxes if global_bboxes else '(none)'}")
        print(f"[INFO] Writing JSONL to: {args.out}")
        print(f"[INFO] Anticipation mode: {is_ant}")

    # Heads present (for metadata) — verb/noun only
    heads_present = _heads_present(model)

    for i, row in enumerate(rows, 1):
        segment_id  = str(row[id_col])
        start_frame = int(row[start_col])
        stop_frame  = int(row[stop_col])

        frames_dir = _frames_dir_for(segment_id, args.frames_base)
        if not os.path.isdir(frames_dir):
            if not args.quiet:
                print(f"[WARN] Missing RGB frames dir → {frames_dir}")
            missing_rgb += 1
            continue

        # Per-segment object frames + per-segment bboxes fallback
        obj_dir = _obj_dir_for(segment_id, args.obj_base)
        seg_bboxes = os.path.join(obj_dir, "bboxes.json") if os.path.isdir(obj_dir) else None

        # Choose which bboxes path to use (global takes priority if present)
        chosen_bboxes = global_bboxes if global_bboxes else (seg_bboxes if seg_bboxes and os.path.isfile(seg_bboxes) else None)

        if not args.quiet:
            print(f"\n▶️ [{i}/{total}] {segment_id}")
            if chosen_bboxes:
                print(f"[BOXES] using: {chosen_bboxes}")
            else:
                print("[BOXES] using: dummy (no bboxes.json found)")

        try:
            # ---------- Inference ----------
            out_raw: Dict[str, Any] = predict_segment(
                frames_dir=frames_dir,
                obj_dir=obj_dir if os.path.isdir(obj_dir) else None,
                model=model,
                cfg=cfg,
                start_frame=start_frame,
                stop_frame=stop_frame,
                bboxes_path=chosen_bboxes,
            )

            # Build a JSONL record (verb/noun only)
            rec: Dict[str, Any] = {
                # IDs
                "segment_id": segment_id,
                "uid": segment_id,           # alias to be extra-robust
                "video_id": _base_id(segment_id),
                # Minimal timing meta (handy for debugging)
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                # Top-5 indices + scores
                "verb_top5_idx":    out_raw.get("verb_top5_idx", []),
                "verb_top5_scores": [float(x) for x in out_raw.get("verb_top5_scores", [])],
                "noun_top5_idx":    out_raw.get("noun_top5_idx", []),
                "noun_top5_scores": [float(x) for x in out_raw.get("noun_top5_scores", [])],
                # Small meta block for traceability
                "meta": {
                    "heads_present": heads_present,
                    "bboxes_path": chosen_bboxes if chosen_bboxes else "",
                },
            }

            _write_jsonl(args.out, rec)

            processed += 1
            if not args.quiet:
                elapsed = time.time() - t0
                v1  = _safe_top1(rec["verb_top5_idx"], -1)
                v1s = _safe_top1(rec["verb_top5_scores"], 0.0)
                n1  = _safe_top1(rec["noun_top5_idx"], -1)
                n1s = _safe_top1(rec["noun_top5_scores"], 0.0)
                print(f"verb@1={v1}({v1s:.3f}) | noun@1={n1}({n1s:.3f})")
                avg = elapsed / max(processed, 1)
                eta = avg * (total - i)
                print(f"⏱  elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

        except Exception as e:
            failures += 1
            print(f"[ERROR] Inference failed for {segment_id}: {e}")
            if not args.quiet:
                tb = "".join(traceback.format_exception_only(type(e), e)).strip()
                print(f"[TRACE] {tb}")
            # Log an error record for alignment
            fail_rec = {
                "segment_id": segment_id,
                "uid": segment_id,
                "video_id": _base_id(segment_id),
                "error": str(e),
                "meta": {
                    "heads_present": heads_present,
                    "bboxes_path": chosen_bboxes if chosen_bboxes else "",
                },
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