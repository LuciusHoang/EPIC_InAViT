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
# Small helpers
# =========================================================

def _parse_frame_num(path: str) -> Optional[int]:
    m = re.search(r'(\d+)\.jpg$', os.path.basename(path))
    return int(m.group(1)) if m else None


def _pil_center_crop_resize(img: Image.Image, target_size: int) -> Image.Image:
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
    if n_in <= 0:
        return [0] * n_out
    if n_out == 1:
        return [0]
    idx = [int(round(i * (n_in - 1) / (n_out - 1))) for i in range(n_out)]
    idx = [min(max(j, 0), n_in - 1) for j in idx]
    while len(idx) < n_out:
        idx.append(idx[-1])
    return idx[:n_out]


def _ten(x: int) -> str:
    return f"{x:010d}"


# =========================================================
# Frame collection & sampling
# =========================================================

def _collect_rgb_frames_in_window(
    frames_dir: str,
    t_star: int,
    window_len: int = 64
) -> Tuple[List[str], List[int]]:
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
# BBox utilities
# =========================================================

def _coords_from_any_box(b: Any) -> Optional[List[float]]:
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return None
    try:
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        return None
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if not (np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1):
        return None
    return [x1, y1, x2, y2]


def _maybe_normalize_224_to_unit(box_xyxy: List[float], crop_size: int) -> Optional[List[float]]:
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
        return None
    return [nx1, ny1, nx2, ny2]


def _derive_segment_key_from_frames_dir(frames_dir: str) -> Optional[str]:
    parts = os.path.normpath(frames_dir).split(os.sep)
    if len(parts) < 2:
        return None
    p = parts[-2]  # P01
    v = parts[-1]  # P01_11
    if re.match(r"^P\d{2}$", p) and re.match(r"^P\d{2}_\d+$", v):
        return f"{p}/{v}"
    return None


def _load_bboxes_json_any(bboxes_path: str) -> Optional[dict]:
    if not bboxes_path or not os.path.isfile(bboxes_path):
        return None
    with open(bboxes_path, "r") as f:
        return json.load(f)


def _extract_segment_dict(boxes_blob: dict, seg_key: Optional[str]) -> Optional[dict]:
    if boxes_blob is None:
        return None
    if isinstance(boxes_blob, dict) and "boxes" in boxes_blob and isinstance(boxes_blob["boxes"], dict):
        if not seg_key:
            return None
        return boxes_blob["boxes"].get(seg_key, None)
    if isinstance(boxes_blob, dict):
        return boxes_blob
    return None


def _pick_top_by_area(boxes_xyxy01: List[List[float]], k: int) -> List[List[float]]:
    if not boxes_xyxy01:
        return []
    with_area = [((b[2]-b[0])*(b[3]-b[1]), b) for b in boxes_xyxy01 if len(b) == 4]
    with_area.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in with_area[:k]]


def _process_box_list(raw_list: list, crop_size: int, limit: int) -> List[List[float]]:
    normed = []
    for b in (raw_list or []):
        xyxy = _coords_from_any_box(b)
        if xyxy is None:
            continue
        out = _maybe_normalize_224_to_unit(xyxy, crop_size)
        if out:
            normed.append(out)
    picked = _pick_top_by_area(normed, limit)
    while len(picked) < limit:
        picked.append([0.0, 0.0, 0.0, 0.0])
    return picked


def build_inavit_boxes(
    per_frame_dict: dict,
    sampled_frame_numbers: List[int],
    crop_size: int,
    O: int,
    U: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(sampled_frame_numbers)
    obj = torch.zeros((1, T, O, 4), dtype=torch.float32)
    hand = torch.zeros((1, T, U, 4), dtype=torch.float32)
    for t, fn in enumerate(sampled_frame_numbers):
        key = _ten(fn)
        entry = per_frame_dict.get(key, {}) if per_frame_dict else {}
        raw_obj  = entry.get("obj", []) or []
        raw_hand = entry.get("hand", []) or []
        objs  = _process_box_list(raw_obj,  crop_size, O)
        hands = _process_box_list(raw_hand, crop_size, U)
        if O > 0:
            obj[0, t, :O] = torch.tensor(objs[:O], dtype=torch.float32)
        if U > 0:
            hand[0, t, :U] = torch.tensor(hands[:U], dtype=torch.float32)
    return obj, hand


# =========================================================
# Output normalization (robust)
# =========================================================

def _cfg_expected_sizes(cfg) -> Dict[str, Optional[int]]:
    """
    Pull expected head sizes from YAML if present:
      MODEL.HEADS.VERB / NOUN / ACTION (Option B), else None.
    """
    sizes = {"verb": None, "noun": None, "action": None}
    if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "HEADS"):
        h = cfg.MODEL.HEADS
        for k in ("VERB", "NOUN", "ACTION"):
            if hasattr(h, k):
                try:
                    sizes[k.lower()] = int(getattr(h, k))
                except Exception:
                    pass
    return sizes


def _guess_verb_noun_by_size(tensors: List[torch.Tensor], expect: Dict[str, Optional[int]]) -> Dict[str, torch.Tensor]:
    """
    Given 2 tensors (order unknown), figure out which is verb vs noun:
      1) use expected sizes if available
      2) else heuristic: smaller dim = verb (97), larger = noun (300)
    """
    res: Dict[str, torch.Tensor] = {}
    if len(tensors) == 1:
        res["action"] = tensors[0]
        return res
    if len(tensors) >= 2:
        a, b = tensors[0], tensors[1]
        aC = a.size(-1)
        bC = b.size(-1)
        vexp, nexp = expect.get("verb"), expect.get("noun")
        # try hard match first
        if vexp and (aC == vexp or bC == vexp):
            res["verb"] = a if aC == vexp else b
            res["noun"] = b if aC == vexp else a
            return res
        if nexp and (aC == nexp or bC == nexp):
            res["noun"] = a if aC == nexp else b
            res["verb"] = b if aC == nexp else a
            return res
        # heuristic
        if aC <= bC:
            res["verb"], res["noun"] = a, b
        else:
            res["verb"], res["noun"] = b, a
        return res
    return res


def _normalize_heads(outputs, cfg=None) -> Dict[str, torch.Tensor]:
    """
    Accept model outputs in many shapes and normalize to a dict of heads.
    Handles:
      - dict with keys among {'verb','noun','action'} or odd names: {'head0','head1','head','head_action',...}
      - tuple/list length 2 or 3 (e.g., (verb,noun[,action]))
      - single tensor (treated as action)
    """
    expect = _cfg_expected_sizes(cfg)

    # If it's a sequence, flatten any nested singletons
    if isinstance(outputs, (list, tuple)):
        flat = []
        for o in outputs:
            if isinstance(o, (list, tuple)) and len(o) == 1 and torch.is_tensor(o[0]):
                flat.append(o[0])
            else:
                flat.append(o)
        outputs = tuple(flat)

    # Case A: direct tensors inside a list/tuple
    if isinstance(outputs, (list, tuple)) and all(torch.is_tensor(x) for x in outputs):
        # try to resolve (verb, noun [, action])
        res = _guess_verb_noun_by_size(list(outputs), expect)
        if len(outputs) == 3:
            # best effort: third is action if size matches or just put it
            third = outputs[2]
            if expect.get("action") and third.size(-1) == expect["action"]:
                res["action"] = third
            elif "action" not in res:
                res["action"] = third
        return res if res else {"action": outputs[0]}

    # Case B: dict-like outputs
    if isinstance(outputs, dict):
        out: Dict[str, torch.Tensor] = {}
        # direct keys
        for k in ("action", "verb", "noun"):
            v = outputs.get(k, None)
            if torch.is_tensor(v):
                out[k] = v
        # odd keys (common in different repos)
        odd_keys = ["head", "head_action", "head0", "head1", "cls_head", "action_head", "verb_head", "noun_head"]
        odd_tensors = [(k, outputs[k]) for k in odd_keys if k in outputs and torch.is_tensor(outputs[k])]
        if odd_tensors:
            # if we already have verb/noun/action don't overwrite; fill missing
            tensors_only = [t for _, t in odd_tensors]
            guess = _guess_verb_noun_by_size(tensors_only, expect)
            for k in ("verb", "noun", "action"):
                if k not in out and k in guess:
                    out[k] = guess[k]
        # final fallback: first tensor found in dict
        if not out:
            for k, v in outputs.items():
                if torch.is_tensor(v):
                    out["action"] = v
                    break
        return out

    # Case C: single tensor
    if torch.is_tensor(outputs):
        return {"action": outputs}

    # Case D: tuple with sub-dicts (rare)
    if isinstance(outputs, (list, tuple)) and any(isinstance(x, dict) for x in outputs):
        # merge recursively
        merged: Dict[str, torch.Tensor] = {}
        for x in outputs:
            if isinstance(x, dict):
                sub = _normalize_heads(x, cfg=cfg)
                merged.update({k: v for k, v in sub.items() if torch.is_tensor(v)})
        if merged:
            return merged

    raise RuntimeError(f"Unsupported model output type: {type(outputs)}")


# =========================================================
# Prediction helpers
# =========================================================

def _topk_from_logits(logits: torch.Tensor, k: int = 5) -> Tuple[List[int], List[float]]:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=1)
    vals, idx = torch.topk(probs, k, dim=1)
    return idx[0].cpu().tolist(), vals[0].cpu().tolist()


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================
# Public API: predict_segment
# =========================================================

@torch.no_grad()
def predict_segment(
    frames_dir: str,
    obj_dir: Optional[str],
    model,
    cfg,
    start_frame: Optional[int] = None,
    stop_frame: Optional[int] = None,
    bboxes_path: Optional[str] = None,
) -> Dict[str, List]:
    """
    InAViT‑style inference for a single EK100 anticipation instance.

    - observe last 64 frames up to t* = start_frame - 1
    - sample 16 frames → 224×224 center crop
    - build (obj + hand) tensors from bboxes.json
    - robustly parse model outputs (tuple/dict/odd head names) to get verb/noun/action
    """
    device = getattr(model, "device", torch.device("cpu"))

    if start_frame is None:
        raise ValueError("predict_segment requires start_frame for anticipation.")
    t_star = max(1, int(start_frame) - 1)

    # 1) frames window
    window_len = 64
    paths_in_window, nums_in_window = _collect_rgb_frames_in_window(frames_dir, t_star, window_len=window_len)
    if len(paths_in_window) == 0:
        raise RuntimeError(f"No frames found in observed window for: {frames_dir}")

    # 2) sample 16 @ 224²
    T_target = int(getattr(cfg.DATA, "NUM_FRAMES", 16))
    crop = int(getattr(cfg.DATA, "TEST_CROP_SIZE", 224))
    mean = getattr(cfg.DATA, "MEAN", [0.5, 0.5, 0.5])
    std  = getattr(cfg.DATA, "STD",  [0.5, 0.5, 0.5])

    clip, sel = _rgb_clip_16x224_from_window(paths_in_window, crop, T_target, mean, std)
    clip = clip.to(device)  # [1,3,T,H,W]
    sampled_nums = [nums_in_window[i] if i < len(nums_in_window) else None for i in sel]
    if any(n is None for n in sampled_nums):
        raise RuntimeError("Sampled frame numbers contain None; check frame naming.")

    # 3) load boxes JSON and build tensors (obj + hand)
    candidate_path = bboxes_path or (os.path.join(obj_dir, "bboxes.json") if obj_dir else None)
    per_frame_dict = None
    if candidate_path:
        blob = _load_bboxes_json_any(candidate_path)
        if blob is not None:
            seg_key = _derive_segment_key_from_frames_dir(frames_dir)
            per_frame_dict = _extract_segment_dict(blob, seg_key)

    if per_frame_dict is None:
        raise FileNotFoundError("No usable bboxes found. Provide --bboxes or per-segment obj_dir/bboxes.json")

    O = int(getattr(cfg.HOIVIT, "O", 5)) if hasattr(cfg, "HOIVIT") else 5
    U = int(getattr(cfg.HOIVIT, "U", 1)) if hasattr(cfg, "HOIVIT") else 1
    obj_boxes, hand_boxes = build_inavit_boxes(
        per_frame_dict=per_frame_dict,
        sampled_frame_numbers=sampled_nums,
        crop_size=crop,
        O=O, U=U,
    )

    # debug
    def _nz(t: torch.Tensor) -> int:
        with torch.no_grad():
            m = (t.abs().sum(dim=-1) > 0)
            return int(m.any(dim=-1).sum().item())
    obj_nz = _nz(obj_boxes) if O > 0 else 0
    hand_nz = _nz(hand_boxes) if U > 0 else 0
    print(f"[DEBUG] InAViT boxes → obj_frames={obj_nz}/{obj_boxes.shape[1]} | hand_frames={hand_nz}/{hand_boxes.shape[1] if U>0 else 0}")
    print(f"[BOXES] source={candidate_path}")

    # Move to device
    obj_boxes = obj_boxes.to(device)
    hand_boxes = hand_boxes.to(device)

    # 4) Build metadata (dict variant first, with aliases)
    meta = {
        "orvit_bboxes": {"obj": obj_boxes, "hand": hand_boxes},
        "hoivit_bboxes": {"obj": obj_boxes, "hand": hand_boxes},
        "boxes": {"obj": obj_boxes, "hand": hand_boxes},
        "bboxes": {"obj": obj_boxes, "hand": hand_boxes},
    }

    # 5) forward (try dict first; fall back to plain obj tensor if needed)
    try:
        raw_outputs = model([clip], meta)
    except Exception as e_dict:
        try:
            raw_outputs = model([clip], {"orvit_bboxes": obj_boxes})
        except Exception as e_alt:
            raise RuntimeError(f"Model forward failed: dict-meta → {e_dict} | obj-only → {e_alt}") from e_alt

    # 6) normalize heads (robust)
    heads = _normalize_heads(raw_outputs, cfg=cfg)
    heads_present = {k: (k in heads and heads[k] is not None) for k in ("action", "verb", "noun")}

    # 7) top‑5
    result: Dict[str, Any] = {
        "verb_top5_idx": [], "verb_top5_scores": [],
        "noun_top5_idx": [], "noun_top5_scores": [],
        "action_top5_idx": [], "action_top5_scores": [],
        "meta": {
            "heads_present": heads_present,
            "bboxes_path": candidate_path,
        }
    }

    if "verb" in heads and heads["verb"] is not None:
        v_idx, v_scr = _topk_from_logits(heads["verb"], k=5)
        result["verb_top5_idx"] = v_idx
        result["verb_top5_scores"] = v_scr

    if "noun" in heads and heads["noun"] is not None:
        n_idx, n_scr = _topk_from_logits(heads["noun"], k=5)
        result["noun_top5_idx"] = n_idx
        result["noun_top5_scores"] = n_scr

    if "action" in heads and heads["action"] is not None:
        a_idx, a_scr = _topk_from_logits(heads["action"], k=5)
        result["action_top5_idx"] = a_idx
        result["action_top5_scores"] = a_scr

    # helpful runtime print (won't spam if you aggregate)
    if not any(result["action_top5_idx"]):
        # no action head detected; reporting verb/noun only is fine
        pass

    return result


# =========================================================
# Build & load model (keep checkpoint heads)
# =========================================================

def build_and_load_model(cfg, pretrained_path: str = "checkpoints/checkpoint_epoch_00081.pyth"):
    """
    Build MotionFormer/InAViT and load weights.
    We KEEP the checkpoint's classifier heads so verb=97, noun=300 stay intact.
    """
    model = build_model(cfg)

    # default_cfg used by load_pretrained (timm-style fields)
    model.default_cfg = {
        "first_conv": "patch_embed.proj",
        "classifier": ["head0", "head1", "head", "head_action"],
        "url": ""
    }

    load_pretrained(
        model=model,
        cfg=model.default_cfg,
        num_classes=cfg.MODEL.NUM_CLASSES,  # ignored when keep_classifier=True (default)
        in_chans=3,
        filter_fn=_conv_filter,
        img_size=cfg.DATA.TEST_CROP_SIZE,
        num_frames=cfg.DATA.NUM_FRAMES,
        pretrained_model=pretrained_path,
        strict=False,
    )

    # Optional: print head structure so you can verify sizes
    try:
        print("[DEBUG] Model heads:")
        if hasattr(model, "head_drop"):
            print("head_drop", model.head_drop)
        for name in ("head", "head0", "head1", "head_action"):
            mod = getattr(model, name, None)
            if mod is not None:
                print(f"{name}", mod)
    except Exception:
        pass

    model.eval()
    device = _best_device()
    model = model.to(device)
    model.device = device
    return model