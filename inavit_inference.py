import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# Add InAViT to the import path
sys.path.append(os.path.abspath("InAViT"))

from slowfast.models.build import build_model
from pretrained_utils import load_pretrained, _conv_filter


# ---------------------------------------------------------
# Utils: frame sampling + clip loading from extracted JPGs
# ---------------------------------------------------------
def _centered_stride_indices(n_frames, num_frames, sampling_rate):
    """
    Prefer sampling with a fixed temporal stride = sampling_rate, centered in the sequence.
    Falls back to uniform sampling if the sequence is too short.
    Always returns length == num_frames (with repeat padding if needed).
    """
    if n_frames <= 0:
        return [0] * num_frames

    window = num_frames * sampling_rate
    if n_frames >= window:
        start = (n_frames - window) // 2
        idx = [start + i * sampling_rate for i in range(num_frames)]
    else:
        # Not enough frames to honor stride window -> uniform over available frames
        idx = np.linspace(0, n_frames - 1, num=num_frames, dtype=int).tolist()

    # Clamp and pad (repeat last) to ensure exactly num_frames indices in range
    idx = [min(max(int(i), 0), n_frames - 1) for i in idx]
    while len(idx) < num_frames:
        idx.append(idx[-1] if idx else 0)
    return idx[:num_frames]


def _pil_center_crop_resize(img, target_size):
    """
    Resize shorter side >= target_size, then center-crop to (target_size, target_size).
    """
    w, h = img.size
    if min(w, h) == 0:
        return img.resize((target_size, target_size), Image.BICUBIC)

    scale = target_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))


def load_rgb_clip_from_dir(frames_dir, cfg):
    """
    Load a clip of shape [1, C, T, H, W] from a directory of extracted RGB frames:
      frames_dir/
        frame_0000000001.jpg
        frame_0000000002.jpg
        ...
    """
    # Find frames
    jpgs = sorted(
        glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
        + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if len(jpgs) == 0:
        raise FileNotFoundError(f"No JPG frames found in: {frames_dir}")

    T = cfg.DATA.NUM_FRAMES
    sr = cfg.DATA.SAMPLING_RATE
    crop = cfg.DATA.TEST_CROP_SIZE

    # Build normalization
    mean = cfg.DATA.MEAN if hasattr(cfg.DATA, "MEAN") else [0.5, 0.5, 0.5]
    std = cfg.DATA.STD if hasattr(cfg.DATA, "STD") else [0.5, 0.5, 0.5]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)

    # Choose indices
    inds = _centered_stride_indices(len(jpgs), T, sr)

    frames = []
    for i in inds:
        img = Image.open(jpgs[i]).convert("RGB")
        img = _pil_center_crop_resize(img, crop)
        tensor = to_tensor(img)          # [C, H, W], 0..1
        tensor = normalize(tensor)       # normalized
        frames.append(tensor)

    clip = torch.stack(frames, dim=1)     # [C, T, H, W]
    clip = clip.unsqueeze(0)              # [1, C, T, H, W]
    return clip


# ---------------------------------------------------------
# Original helpers (kept)
# ---------------------------------------------------------
def preprocess_clip(clip, cfg):
    # Assumes clip is already shaped [1, C, T, H, W] and normalized
    if isinstance(clip, torch.Tensor):
        return clip
    raise TypeError("Input clip must be a torch Tensor")


def load_dummy_bboxes(num_frames, num_obj=4, hand=True):
    # Dummy bounding boxes in expected format [B, T, N_obj, 4]
    obj_boxes = torch.zeros((1, num_frames, num_obj, 4))
    hand_boxes = torch.zeros((1, num_frames, 1, 4)) if hand else None
    return obj_boxes, hand_boxes


# ---------------------------------------------------------
# Inference from EXTRACTED FRAMES (directory), not .mp4/.tar
# ---------------------------------------------------------
def predict_segment(frames_dir, obj_dir, model, cfg):
    """
    frames_dir: path to extracted RGB frames directory (e.g.
      /Volumes/T7_Shield/inference/frames_rgb_flow/rgb/test/PXX/PXX_YY/)
    obj_dir: (optional) path to object/hand crops; unused here (dummy boxes)
    """
    # Load and preprocess frames from a directory of JPGs
    clip = load_rgb_clip_from_dir(frames_dir, cfg)  # [1, C, T, H, W]
    clip = preprocess_clip(clip, cfg)

    device = getattr(model, "device", torch.device("cpu"))
    inputs = [clip.to(device)]

    # Dummy bounding boxes (replace with real ones if desired)
    obj_boxes, hand_boxes = load_dummy_bboxes(cfg.DATA.NUM_FRAMES)
    metadata = {
        "orvit_bboxes": {
            "obj": obj_boxes.to(device),
            "hand": hand_boxes.to(device),
        }
    }

    with torch.no_grad():
        logits = model(inputs, metadata)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return pred_idx.item(), conf.item()


# ---------------------------------------------------------
# Model build + load (CPU / CUDA / MPS aware)
# ---------------------------------------------------------
def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (M1/M2/M3) – use Metal backend if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_and_load_model(cfg, pretrained_path="checkpoints/checkpoint_epoch_00081.pyth"):
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
