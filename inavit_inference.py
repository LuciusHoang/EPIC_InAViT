import torch
import os
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import sys

# Add InAViT to the import path
sys.path.append(os.path.abspath("InAViT"))

from slowfast.models.build import build_model
from pretrained_utils import load_pretrained, _conv_filter
from slowfast.datasets.ek_MF.frame_loader import pack_frames_to_video_clip


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


def predict_segment(video_path, obj_dir, model, cfg):
    # Load and preprocess frames from a video file
    clip = pack_frames_to_video_clip(
        video_path,
        num_frames=cfg.DATA.NUM_FRAMES,
        sampling_rate=cfg.DATA.SAMPLING_RATE
    )

    inputs = [clip.to(model.device)]  # [1, C, T, H, W]

    # Dummy bounding boxes (replace with real ones if desired)
    obj_boxes, hand_boxes = load_dummy_bboxes(cfg.DATA.NUM_FRAMES)
    metadata = {
        "orvit_bboxes": {
            "obj": obj_boxes.to(model.device),
            "hand": hand_boxes.to(model.device),
        }
    }

    with torch.no_grad():
        logits = model(inputs, metadata)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return pred_idx.item(), conf.item()


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
    model = model.cuda() if torch.cuda.is_available() else model
    model.device = next(model.parameters()).device
    return model
