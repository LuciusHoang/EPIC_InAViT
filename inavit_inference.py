#!/usr/bin/env python3
import os
import re
import glob
import json
import sys
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# Make InAViT importable
sys.path.append(os.path.abspath("InAViT"))

from slowfast.models.build import build_model
from pretrained_utils import load_pretrained, _conv_filter


# =========================================================
# Utility helpers
# =========================================================

def _parse_frame_num(path: str) -> Optional[int]:
    """
    Parse trailing integer from frame filename.
    Accepts: frame_0000000001.jpg  OR  0000000001.jpg
    """
    m = re.search(r'(\d+)\.jpg$', os.path.basename(path))
    return int(m.group(1)) if m else None


def _pil_center_crop_resize(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize so the shorter side == target_size, then center-crop to (target_size, target_size).
    """
    w, h = img.size
    if min(w, h) == 0:
        return img.resize((target_size, target_size), Image.BICUBIC)

    scale = target_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))


def _uniform_indices(n_in: int, n_out: int) -> List[int]:
    """
    Uniformly (linearly) sample exactly n_out indices over a sequence of length n_in.
    If n_in < n_out, repeat-pad the last index.
    """
    if n_in <= 0:
        return [0] * n_out
    if n_out == 1:
        return [0]
    idx = [int(round(i * (n_in - 1) / (n_out - 1))) for i in range(n_out)]
    idx = [min(max(j, 0), n_in - 1) for j in idx]
    while len(idx) < n_out:
        idx.append(idx[-1])
    return idx[:n_out]


# =========================================================
# RGB window loader (Ta=1s, 64→16, 224²)
# =========================================================

def _collect_rgb_frames_in_window(
    frames_dir: str,
    t_star: int,
    window_len: int = 64
) -> Tuple[List[str], List[int]]:
    """
    Collect all candidate frame paths in the observed window [t_star - (window_len-1), t_star].
    The directory is expected to contain files named 'frame_%010d.jpg' (preferred)
    or plain '%010d.jpg' (fallback).
    """
    jpgs = sorted(
        glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not jpgs:
        raise FileNotFoundError(f"No JPG frames found in: {frames_dir}")

    items = []
    for p in jpgs:
        n = _parse_frame_num(p)
        if n is not None:
            items.append((n, p))
    if not items:
        raise RuntimeError(f"Could not parse any frame numbers in: {frames_dir}")

    start_n = max(1, t_star - (window_len - 1))
    end_n = t_star

    in_win = [(n, p) for (n, p) in items if (start_n <= n <= end_n)]
    if not in_win:
        in_win = [(n, p) for (n, p) in items if n <= t_star] or items

    in_win.sort(key=lambda x: x[0])
    frame_nums = [n for (n, _) in in_win]
    paths_in_window = [p for (_, p) in in_win]
    return paths_in_window, frame_nums


def _rgb_clip_16x224_from_window(
    paths_in_window: List[str],
    crop_size: int,
    T_target: int,
    mean: List[float],
    std: List[float],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Uniformly sample exactly T_target frames, center-crop to crop_size,
    normalize, and pack into a tensor [1, 3, T, H, W].
    """
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)

    sel = _uniform_indices(len(paths_in_window), T_target)
    frames = []
    for i in sel:
        img = Image.open(paths_in_window[i]).convert("RGB")
        img = _pil_center_crop_resize(img, crop_size)
        frames.append(normalize(to_tensor(img)))  # [C,H,W]

    clip = torch.stack(frames, dim=1).unsqueeze(0)  # [1,3,T,H,W]
    return clip, sel


# =========================================================
# Boxes (hand + objects) loading
# =========================================================

def _ten(x: int) -> str:
    return f"{x:010d}"


def _coords_from_any_box(b: Any) -> Optional[List[float]]:
    """
    Extract [x1,y1,x2,y2] from a box that may be:
      - [x1,y1,x2,y2]
      - [x1,y1,x2,y2,score]
      - tuple variants
    Returns None if invalid.
    """
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        return None
    if not (np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1):
        return None
    return [x1, y1, x2, y2]


def _maybe_normalize_224_to_unit(
    box_xyxy: List[float],
    crop_size: int
) -> List[float]:
    """
    Boxes in your JSON may be in 224×224 pixel space or already normalized.
    If max(abs(coord)) <= 1.5 → assume normalized.
    Else divide by crop_size to get [0,1]. Clamp to [0,1].
    """
    x1, y1, x2, y2 = [float(x) for x in box_xyxy]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    else:
        s = float(crop_size)
        nx1, ny1, nx2, ny2 = x1 / s, y1 / s, x2 / s, y2 / s

    nx1 = min(max(nx1, 0.0), 1.0)
    ny1 = min(max(ny1, 0.0), 1.0)
    nx2 = min(max(nx2, 0.0), 1.0)
    ny2 = min(max(ny2, 0.0), 1.0)

    if nx2 <= nx1 or ny2 <= ny1:
        return []
    return [nx1, ny1, nx2, ny2]


def _derive_segment_key_from_frames_dir(frames_dir: str) -> Optional[str]:
    """
    Try to infer 'PXX/PXX_YY' from a frames directory like:
      .../rgb/P01/P01_11  or  .../object_detection_images/P01/P01_11
    """
    parts = os.path.normpath(frames_dir).split(os.sep)
    if len(parts) < 2:
        return None
    p = parts[-2]  # P01
    v = parts[-1]  # P01_11
    if re.match(r"^P\d{2}$", p) and re.match(r"^P\d{2}_\d+$", v):
        return f"{p}/{v}"
    return None


def _load_bboxes_json_any(bboxes_path: str) -> Optional[dict]:
    """Load JSON; return dict or None."""
    if not bboxes_path or not os.path.isfile(bboxes_path):
        return None
    with open(bboxes_path, "r") as f:
        return json.load(f)


def _extract_segment_dict(
    boxes_blob: dict,
    seg_key: Optional[str]
) -> Optional[dict]:
    """
    From either a flat dict or a nested {'boxes': {seg_key: {...}}} blob,
    return the per-frame dict for this segment.
    """
    if boxes_blob is None:
        return None

    # Nested container?
    if isinstance(boxes_blob, dict) and "boxes" in boxes_blob and isinstance(boxes_blob["boxes"], dict):
        if not seg_key:
            return None
        return boxes_blob["boxes"].get(seg_key, None)

    # Flat → assume already per-frame dict
    if isinstance(boxes_blob, dict):
        return boxes_blob

    return None


def _select_boxes_from_json_for_sampled_frames(
    per_frame_dict: dict,
    sampled_frame_numbers: List[Optional[int]],
    img_size: Tuple[int, int],
    max_obj: int = 4,
    include_hand: bool = True,
    normalize: bool = True,
    propagate_missing: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build tensors from a dict in the format:

      { "0000000001": { "hand": [[...], ...], "obj": [[...], ...] }, ... }

    Returns:
      obj_boxes:  [1, T, max_obj, 4]  (normalized)
      hand_boxes:[1, T, 1, 4] (or None)
    """
    Wc, Hc = img_size  # typically (224, 224)
    T = len(sampled_frame_numbers)

    obj = torch.zeros((1, T, max_obj, 4), dtype=torch.float32)
    hand = torch.zeros((1, T, 1, 4), dtype=torch.float32) if include_hand else None

    last_hand_box: Optional[List[float]] = None
    last_obj_boxes: Optional[List[List[float]]] = None

    for t, fn in enumerate(sampled_frame_numbers):
        key = _ten(fn) if fn is not None else None
        entry = per_frame_dict.get(key, {}) if (per_frame_dict is not None and key is not None) else {}

        # ---- HAND ----
        hand_list = entry.get("hand", []) or []
        hand_box_norm: Optional[List[float]] = None
        # hand_list may be list-of-lists or (incorrectly) a single flat list
        if hand_list and isinstance(hand_list[0], (int, float)):
            hand_list = [hand_list]  # wrap flat → list-of-lists

        if hand_list:
            cand = []
            areas = []
            for b in hand_list:
                bx4 = _coords_from_any_box(b)
                if bx4 is None:
                    continue
                bx = _maybe_normalize_224_to_unit(bx4, Hc if normalize else 1)
                if not bx:
                    continue
                cand.append(bx)
                areas.append((bx[2] - bx[0]) * (bx[3] - bx[1]))
            if cand:
                idx = int(np.argmax(areas))
                hand_box_norm = cand[idx]

        # ---- OBJECTS ----
        obj_list = entry.get("obj", []) or []
        obj_boxes_norm: List[List[float]] = []
        if obj_list:
            cand = []
            for b in obj_list:
                bx4 = _coords_from_any_box(b)
                if bx4 is None:
                    continue
                bx = _maybe_normalize_224_to_unit(bx4, Wc if normalize else 1)
                if not bx:
                    continue
                cand.append(bx)
            if cand:
                cand.sort(key=lambda bb: (bb[2]-bb[0]) * (bb[3]-bb[1]), reverse=True)
                obj_boxes_norm = cand[:max_obj]

        # ---- Propagate if missing ----
        if propagate_missing:
            if hand_box_norm is None and last_hand_box is not None:
                hand_box_norm = last_hand_box
            if (not obj_boxes_norm) and (last_obj_boxes is not None):
                obj_boxes_norm = last_obj_boxes

        # ---- Write tensors ----
        if include_hand and hand_box_norm is not None:
            hand[0, t, 0] = torch.tensor(hand_box_norm, dtype=torch.float32)

        for k, bx in enumerate(obj_boxes_norm[:max_obj]):
            obj[0, t, k] = torch.tensor(bx, dtype=torch.float32)

        # Update last seen (only if present this step)
        if hand_box_norm is not None:
            last_hand_box = hand_box_norm
        if obj_boxes_norm:
            last_obj_boxes = obj_boxes_norm

    return obj, hand


def _dummy_bboxes(T: int, max_obj: int = 4, include_hand: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    obj = torch.zeros((1, T, max_obj, 4), dtype=torch.float32)
    hand = torch.zeros((1, T, 1, 4), dtype=torch.float32) if include_hand else None
    return obj, hand


# =========================================================
# Top‑k helpers
# =========================================================

def _topk_from_logits(logits: torch.Tensor, k: int = 5) -> Tuple[List[int], List[float]]:
    """
    Softmax → top‑k indices and scores (lists).
    logits: [1, C] or [C]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=1)
    vals, idx = torch.topk(probs, k, dim=1)
    return idx[0].cpu().tolist(), vals[0].cpu().tolist()


# =========================================================
# Output normalization (single‑head or multi‑head)
# =========================================================

def _normalize_heads(outputs) -> Dict[str, torch.Tensor]:
    """
    Accept model outputs in several shapes and normalize to a dict of heads.

    Supports:
      - dict with any subset of {'verb','noun','action'}
      - tensor or 1D/2D logits => interpreted as action head
      - tuple/list with one element => unpack
    """
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        outputs = outputs[0]

    if isinstance(outputs, dict):
        out = {}
        if 'action' in outputs:
            out['action'] = outputs['action']
        if 'verb' in outputs:
            out['verb'] = outputs['verb']
        if 'noun' in outputs:
            out['noun'] = outputs['noun']
        if not out:
            for k in outputs:
                v = outputs[k]
                if torch.is_tensor(v):
                    out['action'] = v
                    break
        return out

    if torch.is_tensor(outputs):
        return {'action': outputs}

    raise RuntimeError(f"Unsupported model output type: {type(outputs)}")


# =========================================================
# Public API: predict_segment (RGB‑only; Ta=1s, 64→16)
# =========================================================

@torch.no_grad()
def predict_segment(
    frames_dir: str,
    obj_dir: Optional[str],
    model,
    cfg,
    start_frame: Optional[int] = None,
    stop_frame: Optional[int] = None,
    bboxes_path: Optional[str] = None,   # <— NEW: direct path to bboxes.json
) -> Dict[str, List]:
    """
    InAViT‑style inference for a single EK100 anticipation instance.

    Protocol:
      - t* = start_frame - 1  (anticipation gap Ta=1s)
      - observe last 64 frames up to t*
      - uniformly sample 16 frames → 224×224 center crop
      - build 1 hand + up to 4 object boxes per frame (from bboxes.json if present; else dummy zeros)
      - forward model and return Top‑5.
    """
    device = getattr(model, "device", torch.device("cpu"))

    # ---- 0) derive t* ----
    if start_frame is None:
        raise ValueError("predict_segment requires start_frame for anticipation.")
    t_star = max(1, int(start_frame) - 1)

    # ---- 1) collect window frames [t*-63, t*] ----
    window_len = 64
    paths_in_window, nums_in_window = _collect_rgb_frames_in_window(frames_dir, t_star, window_len=window_len)
    if len(paths_in_window) == 0:
        raise RuntimeError(f"No frames found in observed window for: {frames_dir}")

    # ---- 2) sample 16 @ 224² ----
    T_target = int(getattr(cfg.DATA, "NUM_FRAMES", 16))
    crop = int(getattr(cfg.DATA, "TEST_CROP_SIZE", 224))
    mean = getattr(cfg.DATA, "MEAN", [0.5, 0.5, 0.5])
    std  = getattr(cfg.DATA, "STD",  [0.5, 0.5, 0.5])

    clip, sel = _rgb_clip_16x224_from_window(paths_in_window, crop, T_target, mean, std)
    clip = clip.to(device)  # [1,3,T,H,W]

    # Selected *frame numbers* for boxes lookup
    sampled_nums = [nums_in_window[i] if i < len(nums_in_window) else None for i in sel]

    # ---- 3) Load boxes (hand + up to 4 objects)
    # Preferred: explicit bboxes_path; Fallback: obj_dir/bboxes.json
    candidate_path = bboxes_path or (os.path.join(obj_dir, "bboxes.json") if obj_dir else None)
    per_frame_dict = None
    if candidate_path:
        blob = _load_bboxes_json_any(candidate_path)
        if blob is not None:
            seg_key = _derive_segment_key_from_frames_dir(frames_dir)
            per_frame_dict = _extract_segment_dict(blob, seg_key)

    if per_frame_dict is not None and all(n is not None for n in sampled_nums):
        obj_boxes, hand_boxes = _select_boxes_from_json_for_sampled_frames(
            per_frame_dict,
            sampled_frame_numbers=sampled_nums,
            img_size=(crop, crop),
            max_obj=4,
            include_hand=True,
            normalize=True,
            propagate_missing=True,
        )
    else:
        obj_boxes, hand_boxes = _dummy_bboxes(T=T_target, max_obj=4, include_hand=True)

    # ---- Debug counts ----
    def _nz(t: torch.Tensor) -> int:
        with torch.no_grad():
            m = (t.abs().sum(dim=-1) > 0)  # [1,T,max_obj] or [1,T,1]
            return int(m.any(dim=-1).sum().item())
    obj_nz = _nz(obj_boxes)
    hand_nz = _nz(hand_boxes) if hand_boxes is not None else 0
    print(f"[DEBUG] boxes used → obj_frames_with_boxes={obj_nz}/{obj_boxes.shape[1]} | hand_frames_with_boxes={hand_nz}/{obj_boxes.shape[1]}")
    src_note = candidate_path if candidate_path else "dummy"
    print(f"[BOXES] source={src_note}")

    metadata = {
        "orvit_bboxes": {"obj": obj_boxes.to(device), "hand": hand_boxes.to(device) if hand_boxes is not None else None},
        "hoivit_bboxes": {"obj": obj_boxes.to(device), "hand": hand_boxes.to(device) if hand_boxes is not None else None},
        "bboxes": {"obj": obj_boxes.to(device), "hand": hand_boxes.to(device) if hand_boxes is not None else None},
        "boxes": {"obj": obj_boxes.to(device), "hand": hand_boxes.to(device) if hand_boxes is not None else None},
    }

    # ---- 4) forward ----
    raw_outputs = model([clip], metadata)
    heads = _normalize_heads(raw_outputs)

    # ---- 5) top‑5 per available head ----
    result = {
        "verb_top5_idx": [],
        "verb_top5_scores": [],
        "noun_top5_idx": [],
        "noun_top5_scores": [],
        "action_top5_idx": [],
        "action_top5_scores": [],
    }

    if "action" in heads and heads["action"] is not None:
        a_idx, a_scr = _topk_from_logits(heads["action"], k=5)
        result["action_top5_idx"] = a_idx
        result["action_top5_scores"] = a_scr

    if "verb" in heads and heads["verb"] is not None:
        v_idx, v_scr = _topk_from_logits(heads["verb"], k=5)
        result["verb_top5_idx"] = v_idx
        result["verb_top5_scores"] = v_scr

    if "noun" in heads and heads["noun"] is not None:
        n_idx, n_scr = _topk_from_logits(heads["noun"], k=5)
        result["noun_top5_idx"] = n_idx
        result["noun_top5_scores"] = n_scr

    return result


# =========================================================
# Model build + load (device‑agnostic; CPU/MPS/CUDA)
# =========================================================

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_and_load_model(cfg, pretrained_path: str = "checkpoints/checkpoint_epoch_00081.pyth"):
    """
    Build the InAViT (MotionFormer backbone + HOI modules) and load weights.
    Leaves classifier heads intact (e.g., 3806 for EK100).
    """
    model = build_model(cfg)

    model.default_cfg = {
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        "url": ""
    }

    load_pretrained(
        model=model,
        cfg=model.default_cfg,
        num_classes=cfg.MODEL.NUM_CLASSES,
        in_chans=3,
        filter_fn=_conv_filter,
        img_size=cfg.DATA.TEST_CROP_SIZE,
        num_frames=cfg.DATA.NUM_FRAMES,
        pretrained_model=pretrained_path,
        strict=False,
    )

    model.eval()
    device = _best_device()
    model = model.to(device)
    model.device = device
    return model