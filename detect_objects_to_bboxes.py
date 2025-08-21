#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# -----------------------------
# Parsing / utilities
# -----------------------------
def parse_frame_num(path: str) -> Optional[int]:
    """
    Accepts 'frame_0000000123.jpg' or '0000000123.jpg'
    """
    m = re.search(r'(\d+)\.jpe?g$', os.path.basename(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def ten(n: int) -> str:
    return f"{n:010d}"

def log(msg: str):
    print(f"[LOG] {msg}")

# -----------------------------
# 224 mapping (resize short side -> 224, center-crop 224x224)
# -----------------------------
CROP_SIZE = 224

def center_crop_params_after_short_side_resize(src_w: int, src_h: int, target: int = CROP_SIZE) -> Tuple[float, int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1.0, 0, 0
    short = min(src_w, src_h)
    scale = target / float(short)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    return scale, left, top

def map_src_box_to_center_crop_224(box_xyxy: List[float], src_w: int, src_h: int) -> Optional[List[float]]:
    """
    Box is in ORIGINAL image coords; map it through the same transform used by InAViT inference.
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

# -----------------------------
# Detector (CPU)
# -----------------------------
def build_detector(device: torch.device):
    """
    TorchVision Faster R-CNN (COCO) — solid & relatively light on CPU for just a folder.
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(device)
    return model

def run_detector_on_image(model, device, img: Image.Image, thr: float = 0.5, max_obj: int = 4) -> List[List[float]]:
    """
    Returns a list of [x1,y1,x2,y2] in SOURCE image coords.
    """
    to_tensor = transforms.ToTensor()
    with torch.no_grad():
        pred = model([to_tensor(img).to(device)])[0]
    boxes = pred.get("boxes", torch.empty(0)).cpu().numpy()
    scores = pred.get("scores", torch.empty(0)).cpu().numpy()
    keep = (scores >= float(thr))
    boxes = boxes[keep]
    scores = scores[keep]
    # sort by score desc & limit
    if len(scores) > 0:
        order = np.argsort(-scores)
        boxes = boxes[order][:max_obj]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
    # ensure xyxy floats
    out = []
    for b in boxes:
        x1, y1, x2, y2 = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out

# -----------------------------
# Fill missing frames (forward-fill)
# -----------------------------
def forward_fill_boxes(flat: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    If frames are missing in sequence, copy the last seen 'obj' list forward.
    Adds missing keys based on numeric range.
    """
    if not flat:
        return flat
    # derive integer keys
    items = []
    for k in flat.keys():
        try:
            items.append(int(k))
        except Exception:
            pass
    if not items:
        return flat
    items_sorted = sorted(items)
    start, end = items_sorted[0], items_sorted[-1]
    last_objs = None
    for n in range(start, end + 1):
        k = ten(n)
        if k in flat:
            if flat[k].get("obj"):
                last_objs = flat[k]["obj"]
            else:
                if last_objs is not None:
                    flat[k]["obj"] = last_objs
        else:
            flat[k] = {"hand": [[]], "obj": last_objs or []}
    return flat

# -----------------------------
# Main pipeline
# -----------------------------
def process_folder(
    images_dir: str,
    out_path: str,
    thr: float = 0.5,
    max_obj: int = 4,
    fill_missing: bool = True,
    log_every: int = 50,
    sample_log: bool = False
):
    device = torch.device("cpu")
    model = build_detector(device)

    jpgs = sorted(
        glob.glob(os.path.join(images_dir, "frame_*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )
    if not jpgs:
        raise FileNotFoundError(f"No JPGs found in {images_dir}")

    total = len(jpgs)
    log(f"Found {total} frames in {images_dir}")
    t0 = time.time()

    results: Dict[str, Dict[str, Any]] = {}

    for idx, p in enumerate(jpgs, 1):
        n = parse_frame_num(p)
        if n is None:
            continue
        key = ten(n)

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log(f"Skip unreadable frame: {p} ({e})")
            continue

        src_w, src_h = img.size

        # Detect in SOURCE coords
        boxes_src = run_detector_on_image(model, device, img, thr=thr, max_obj=max_obj)

        # Map each to 224×224 crop coords
        boxes_224: List[List[float]] = []
        for b in boxes_src:
            b224 = map_src_box_to_center_crop_224(b, src_w, src_h)
            if b224 is not None:
                boxes_224.append(b224)

        results[key] = {
            "hand": [[]],       # only objects requested
            "obj": boxes_224  # list of [x1,y1,x2,y2] in 224 space
        }

        # -------- progress log --------
        if idx % log_every == 0 or idx == total:
            nonempty = sum(1 for v in results.values() if v.get("obj"))
            elapsed = time.time() - t0
            fps = idx / max(elapsed, 1e-6)
            msg = f"Processed {idx}/{total} frames ({nonempty} with detections) | {fps:.2f} fps"
            if sample_log and boxes_224:
                msg += f" | sample={boxes_224[0]}"
            log(msg)

    if fill_missing:
        log("Forward-filling missing frames...")
        results = forward_fill_boxes(results)

    # Save flat dict
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Final summary
    total_out = len(results)
    nonempty = sum(1 for v in results.values() if v.get("obj"))
    elapsed = time.time() - t0
    log(f"✅ Wrote: {out_path}")
    log(f"[STATS] frames_out={total_out} | frames_with_objs={nonempty} | thr={thr} | max_obj={max_obj} | time={elapsed:.1f}s")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Detect objects and export flat bboxes.json for InAViT.")
    ap.add_argument("--images-dir", required=True,
                    help="Folder with frames (e.g. /Volumes/T7_Shield/inference/object_detection_images/P01/P01_11)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <images-dir>/bboxes.json)")
    ap.add_argument("--thr", type=float, default=0.5, help="Score threshold")
    ap.add_argument("--max-obj", type=int, default=4, help="Max objects per frame")
    ap.add_argument("--no-fill-missing", action="store_true",
                    help="Disable forward-filling for missing frames")
    ap.add_argument("--log-every", type=int, default=50, help="Log progress every N frames")
    ap.add_argument("--sample-log", action="store_true", help="Include one sample box in progress logs")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.images_dir, "bboxes.json")
    process_folder(
        images_dir=args.images_dir,
        out_path=out_path,
        thr=args.thr,
        max_obj=args.max_obj,
        fill_missing=(not args.no_fill_missing),
        log_every=args.log_every,
        sample_log=args.sample_log,
    )

if __name__ == "__main__":
    main()