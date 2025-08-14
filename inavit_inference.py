import os
import re
import glob
import json
import sys
from typing import List, Optional, Tuple, Dict

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
    # Safety clamps
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

    Returns:
      paths_in_window: ordered list of file paths within the window (by frame number)
      frame_nums:      corresponding integer frame numbers (same length as paths)
    """
    # Gather all jpgs once (support both patterns)
    jpgs = sorted(
        glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not jpgs:
        raise FileNotFoundError(f"No JPG frames found in: {frames_dir}")

    # Map to numbers
    items = []
    for p in jpgs:
        n = _parse_frame_num(p)
        if n is not None:
            items.append((n, p))
    if not items:
        raise RuntimeError(f"Could not parse any frame numbers in: {frames_dir}")

    # Window range
    start_n = max(1, t_star - (window_len - 1))
    end_n = t_star

    # Pick those inside the window
    in_win = [(n, p) for (n, p) in items if (start_n <= n <= end_n)]
    if not in_win:
        # If missing, fall back to "best we can": use all <= t_star, or the whole list
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
    From the observed window, uniformly sample exactly T_target frames, center-crop to crop_size,
    normalize, and pack into a tensor [1, 3, T, H, W].
    Returns:
      clip:  [1,3,T,H,W]
      sel:   selected *indices into paths_in_window* (length T_target)
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

def _load_bboxes_json(json_path: str) -> Optional[dict]:
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def _ten(x: int) -> str:
    return f"{x:010d}"


def _select_boxes_from_json_for_sampled_frames(
    data: dict,
    sampled_frame_numbers: List[Optional[int]],
    img_size: Tuple[int, int],
    max_obj: int = 4,
    include_hand: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build:
      obj_boxes:  [1, T, max_obj, 4]  (x1,y1,x2,y2)
      hand_boxes: [1, T, 1, 4] (or None)
    Coordinates normalized to [0,1] w.r.t. network input size (img_size) if normalize=True.
    JSON format expected per frame key:
      { "0000000001": {"obj": [[x1,y1,x2,y2],...], "hand": [[...], ...]}, ... }
    """
    Wc, Hc = img_size
    T = len(sampled_frame_numbers)

    obj = torch.zeros((1, T, max_obj, 4), dtype=torch.float32)
    hand = torch.zeros((1, T, 1, 4), dtype=torch.float32) if include_hand else None

    for t, fn in enumerate(sampled_frame_numbers):
        key = _ten(fn) if fn is not None else None
        entry = data.get(key, {}) if (data is not None and key is not None) else {}
        obj_list = entry.get("obj", []) or []
        hand_list = entry.get("hand", []) or []

        # Objects (up to max_obj)
        for k, box in enumerate(obj_list[:max_obj]):
            x1, y1, x2, y2 = box
            if normalize:
                obj[0, t, k, 0] = x1 / float(Wc)
                obj[0, t, k, 1] = y1 / float(Hc)
                obj[0, t, k, 2] = x2 / float(Wc)
                obj[0, t, k, 3] = y2 / float(Hc)
            else:
                obj[0, t, k] = torch.tensor([x1, y1, x2, y2])

        # Hand: pick the largest area if multiple
        if include_hand and hand_list:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in hand_list]
            idx = int(np.argmax(areas))
            x1, y1, x2, y2 = hand_list[idx]
            if normalize:
                hand[0, t, 0, 0] = x1 / float(Wc)
                hand[0, t, 0, 1] = y1 / float(Hc)
                hand[0, t, 0, 2] = x2 / float(Wc)
                hand[0, t, 0, 3] = y2 / float(Hc)
            else:
                hand[0, t, 0] = torch.tensor([x1, y1, x2, y2])

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
    # Unwrap list/tuple
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        outputs = outputs[0]

    # Dict
    if isinstance(outputs, dict):
        out = {}
        if 'action' in outputs:
            out['action'] = outputs['action']
        if 'verb' in outputs:
            out['verb'] = outputs['verb']
        if 'noun' in outputs:
            out['noun'] = outputs['noun']
        # If dict but no known keys, try to guess a single tensor as action
        if not out:
            # Try common single key
            for k in outputs:
                v = outputs[k]
                if torch.is_tensor(v):
                    out['action'] = v
                    break
        return out

    # Tensor => action logits
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
) -> Dict[str, List]:
    """
    InAViT‑style inference for a single EK100 anticipation instance.

    Protocol:
      - t* = start_frame - 1  (anticipation gap Ta=1s)
      - observe last 64 frames up to t*
      - uniformly sample 16 frames → 224×224 center crop
      - build 1 hand + up to 4 object boxes per frame (from bboxes.json if present; else dummy zeros)
      - forward model and return Top‑5. If only action head exists (anticipation), verb/noun lists are empty.
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

    # selected *frame numbers* (for boxes lookup)
    sampled_nums = [nums_in_window[i] if i < len(nums_in_window) else None for i in sel]

    # ---- 3) boxes (hand + up to 4 objects) ----
    obj_boxes, hand_boxes = None, None
    if obj_dir and os.path.isdir(obj_dir):
        data = _load_bboxes_json(os.path.join(obj_dir, "bboxes.json"))
        if data is not None and all(n is not None for n in sampled_nums):
            obj_boxes, hand_boxes = _select_boxes_from_json_for_sampled_frames(
                data=data,
                sampled_frame_numbers=sampled_nums,
                img_size=(crop, crop),
                max_obj=4,
                include_hand=True,
                normalize=True,
            )

    if obj_boxes is None:
        # Fallback to dummy (zeros) with correct shapes
        obj_boxes, hand_boxes = _dummy_bboxes(T=T_target, max_obj=4, include_hand=True)

    metadata = {
        # Name kept to match existing model wrappers in your repo
        "orvit_bboxes": {
            "obj": obj_boxes.to(device),                   # [1,T,4,4]
            "hand": hand_boxes.to(device) if hand_boxes is not None else None,  # [1,T,1,4]
        }
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
    Leaves classifier heads intact (97 / 300 / 3805 or 3806) — do NOT remap for subsets.
    """
    model = build_model(cfg)

    # Hints for the loader (kept from your previous version)
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