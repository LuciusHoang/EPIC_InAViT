#!/usr/bin/env python3
"""
Hand-only detector (no deep weights): skin-color + optional motion prior.
Outputs flat JSON that InAViT expects, i.e.:

{
  "0000000001": { "hand": [[x1,y1,x2,y2]], "obj": [[]] },
  "0000000002": { "hand": [],               "obj": [[]] },
  ...
}

All hand boxes are given in the 224×224 center-crop coordinate space used
by your InAViT inference (short-side resize to 224, then center crop).

Mac/CPU friendly. Dependencies: opencv-python, numpy, pillow.
"""
import os
import re
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[LOG] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def err(msg: str) -> None:
    print(f"[ERROR] {msg}")

# -----------------------------------------------------------------------------
# Frame utilities
# -----------------------------------------------------------------------------
def parse_frame_num(path: str) -> Optional[int]:
    """
    Accepts 'frame_0000000123.jpg' or '0000000123.jpg'
    """
    m = re.search(r'(\d+)\.jpe?g$', os.path.basename(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def ten(n: int) -> str:
    return f"{n:010d}"

# -----------------------------------------------------------------------------
# 224 mapping (resize short side -> 224, center-crop 224×224)
# -----------------------------------------------------------------------------
CROP_SIZE = 224

def center_crop_params_after_short_side_resize(src_w: int, src_h: int, target: int = CROP_SIZE) -> Tuple[float, int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1.0, 0, 0
    short = min(src_w, src_h)
    scale = target / float(short)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    left = (new_w - target) // 2
    top  = (new_h - target) // 2
    return scale, left, top

def map_src_box_to_center_crop_224(box_xyxy: List[float], src_w: int, src_h: int) -> Optional[List[float]]:
    """
    Box is in ORIGINAL image coords; map it through the same transform used by InAViT inference.
    Returns [x1,y1,x2,y2] in 224×224 space or None if invalid.
    """
    if len(box_xyxy) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    if not (x2 > x1 and y2 > y1):
        return None

    scale, left, top = center_crop_params_after_short_side_resize(src_w, src_h, CROP_SIZE)
    # resize
    x1r, y1r = x1 * scale, y1 * scale
    x2r, y2r = x2 * scale, y2 * scale
    # center crop
    x1c, y1c = x1r - left, y1r - top
    x2c, y2c = x2r - left, y2r - top
    # clip to crop window
    x1c = max(0.0, min(CROP_SIZE, x1c))
    y1c = max(0.0, min(CROP_SIZE, y1c))
    x2c = max(0.0, min(CROP_SIZE, x2c))
    y2c = max(0.0, min(CROP_SIZE, y2c))
    if x2c <= x1c or y2c <= y1c:
        return None
    return [float(x1c), float(y1c), float(x2c), float(y2c)]

# -----------------------------------------------------------------------------
# Skin-color + motion hand detector (no weights)
# -----------------------------------------------------------------------------
def skin_mask(img_bgr: np.ndarray,
              use_hsv: bool = True,
              use_ycrcb: bool = True) -> np.ndarray:
    """
    Heuristic skin segmentation combining HSV and YCrCb bands.
    Tuned to be permissive; you can adjust ranges with CLI.
    """
    masks = []
    if use_hsv:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Typical skin-ish ranges (broad). H in [0, 25/30] or [0, 17°], S >= 40, V >= 40
        lower1 = np.array([0,   40, 40], dtype=np.uint8)
        upper1 = np.array([25, 255,255], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        masks.append(m1)
    if use_ycrcb:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        # Cr in [135, 180], Cb in [85, 135] — classic heuristic
        lower2 = np.array([0, 135,  85], dtype=np.uint8)
        upper2 = np.array([255,180,135], dtype=np.uint8)
        m2 = cv2.inRange(ycrcb, lower2, upper2)
        masks.append(m2)
    if not masks:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_and(mask, m)
    return mask

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological cleanups to remove noise and fill holes.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

def add_motion_prior(mask: np.ndarray,
                     fgmask: Optional[np.ndarray],
                     alpha: float = 0.5) -> np.ndarray:
    """
    Boost skin mask where motion is present (MOG2 output). alpha∈[0,1].
    """
    if fgmask is None:
        return mask
    m = cv2.normalize(fgmask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    s = (mask.astype(np.float32) / 255.0)
    boosted = np.clip((1 - alpha) * s + alpha * (s * m * 2.0), 0, 1)
    return (boosted * 255).astype(np.uint8)

def largest_hand_like_contour(mask: np.ndarray,
                              min_area: int = 500,
                              max_aspect: float = 2.0,
                              min_solidity: float = 0.5) -> Optional[List[float]]:
    """
    Return XYXY of the best hand-like blob in the mask.
    Filters by area, aspect ratio, and solidity to reduce false positives.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > max_aspect:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / max(1.0, hull_area)
        if solidity < min_solidity:
            continue
        # score: prefer bigger + squarer + more solid
        score = (area) * (1.5 - min(aspect, 1.5)) * (0.5 + 0.5 * solidity)
        if score > best_score:
            best_score = score
            best = [float(x), float(y), float(x + w), float(y + h)]
    return best

# -----------------------------------------------------------------------------
# Forward-fill for missing frames (and empty detections)
# -----------------------------------------------------------------------------
def forward_fill_boxes(flat: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create truly missing frames in the numeric range and copy last seen 'hand'.
    If an existing frame has empty 'hand', fill it from the last seen.
    Always keep "obj": [[]] for compatibility.
    """
    if not flat:
        return flat

    keys_int = []
    for k in flat.keys():
        try:
            keys_int.append(int(k))
        except Exception:
            pass
    if not keys_int:
        return flat

    keys_int.sort()
    start, end = keys_int[0], keys_int[-1]
    last_hand = None

    for n in range(start, end + 1):
        k = ten(n)
        if k in flat:
            hand_list = flat[k].get("hand", [])
            if hand_list:
                last_hand = hand_list
            else:
                flat[k]["hand"] = last_hand or []
            if "obj" not in flat[k]:
                flat[k]["obj"] = [[]]
        else:
            flat[k] = {"hand": (last_hand or []), "obj": [[]]}

    return flat

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def process_folder(
    images_dir: str,
    out_path: str,
    use_motion: bool = True,
    skin_min_area: int = 500,
    max_aspect: float = 2.0,
    min_solidity: float = 0.5,
    motion_alpha: float = 0.5,
    fill_missing: bool = True,
    log_every: int = 50,
    sample_log: bool = False,
    debug_dir: Optional[str] = None
):
    # enumerate frames
    jpgs = sorted(
        glob.glob(os.path.join(images_dir, "frame_*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )
    if not jpgs:
        raise FileNotFoundError(f"No JPGs found in {images_dir}")

    total = len(jpgs)
    log(f"Found {total} frames in {images_dir}")

    # optional motion prior
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False) if use_motion else None

    results: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    processed = 0
    has_hand = 0

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for idx, path in enumerate(jpgs, 1):
        n = parse_frame_num(path)
        if n is None:
            continue
        key = ten(n)

        # Load image
        try:
            pil = Image.open(path).convert("RGB")
            src_w, src_h = pil.size
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            warn(f"Skipping unreadable image: {path} ({e})")
            continue

        # Build skin mask
        mask = skin_mask(bgr, use_hsv=True, use_ycrcb=True)
        mask = refine_mask(mask)

        # Motion prior
        if mog2 is not None:
            fg = mog2.apply(bgr)
            mask = add_motion_prior(mask, fg, alpha=motion_alpha)
            # binarize again
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Pick best hand-like blob
        bbox_src = largest_hand_like_contour(
            mask,
            min_area=skin_min_area,
            max_aspect=max_aspect,
            min_solidity=min_solidity
        )

        hand_224: List[List[float]] = []
        if bbox_src is not None:
            mapped = map_src_box_to_center_crop_224(bbox_src, src_w, src_h)
            if mapped is not None:
                hand_224 = [mapped]
                has_hand += 1

        results[key] = {
            "hand": hand_224,   # list (possibly empty) of [x1,y1,x2,y2] in 224 space
            "obj": [[]]          # keep present but empty for compatibility
        }

        # Optional debug overlay
        if debug_dir:
            vis = bgr.copy()
            if bbox_src is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox_src]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"dbg_{key}.jpg"), vis)

        processed += 1
        if (idx % log_every == 0) or (idx == total):
            elapsed = time.time() - t0
            fps = processed / max(elapsed, 1e-6)
            smpl = f" | sample_hand={hand_224[0]}" if (sample_log and hand_224) else ""
            log(f"Processed {idx}/{total} | frames_with_hand={has_hand} | {fps:.2f} fps{smpl}")

    # Forward-fill
    if fill_missing:
        log("Forward-filling missing/empty hand frames…")
        results = forward_fill_boxes(results)

    # Write JSON
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total_out = len(results)
    nonempty = sum(1 for v in results.values() if v.get("hand"))
    elapsed = time.time() - t0
    log(f"✅ Wrote: {out_path}")
    log(f"[STATS] frames_in={total} | frames_out={total_out} | frames_with_hand={nonempty} | time={elapsed:.1f}s")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Hand-only detector (skin+motion), export flat bboxes.json for InAViT.")
    ap.add_argument("--images-dir", required=True,
                    help="Folder with frames (e.g. /Volumes/T7_Shield/inference/object_detection_images/P01/P01_11)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <images-dir>/bboxes_hand.json)")
    ap.add_argument("--no-motion", action="store_true",
                    help="Disable motion prior (MOG2).")
    ap.add_argument("--skin-min-area", type=int, default=500,
                    help="Minimum contour area to accept as hand.")
    ap.add_argument("--max-aspect", type=float, default=2.0,
                    help="Max aspect ratio (long/short side).")
    ap.add_argument("--min-solidity", type=float, default=0.5,
                    help="Min solidity (area/convex hull area).")
    ap.add_argument("--motion-alpha", type=float, default=0.5,
                    help="Blend factor for motion prior (0..1).")
    ap.add_argument("--no-fill-missing", action="store_true",
                    help="Disable forward-filling for missing/empty frames.")
    ap.add_argument("--log-every", type=int, default=50,
                    help="Log progress every N frames.")
    ap.add_argument("--sample-log", action="store_true",
                    help="Include a sample hand box in progress logs.")
    ap.add_argument("--debug-dir", default=None,
                    help="Optional folder to save debug overlay images.")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.images_dir, "bboxes_hand.json")
    process_folder(
        images_dir=args.images_dir,
        out_path=out_path,
        use_motion=(not args.no_motion),
        skin_min_area=args.skin_min_area,
        max_aspect=args.max_aspect,
        min_solidity=args.min_solidity,
        motion_alpha=args.motion_alpha,
        fill_missing=(not args.no_fill_missing),
        log_every=args.log_every,
        sample_log=args.sample_log,
        debug_dir=args.debug_dir
    )

if __name__ == "__main__":
    main()