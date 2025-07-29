import torch
import os
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys

sys.path.append(os.path.abspath("InAViT"))
from slowfast.models.build import build_model
from pretrained_utils import load_pretrained, _conv_filter


def preprocess_frames(frames, cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.DATA.TEST_CROP_SIZE),
        transforms.CenterCrop(cfg.DATA.TEST_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])
    processed = [
        transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        for f in frames
    ]
    clip = torch.stack(processed, dim=1)  # [C, T, H, W]
    return clip


def load_frames_from_folder(frame_dir, num_frames=16):
    frame_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    # Uniformly sample frames
    idx = np.linspace(0, len(frame_paths) - 1, num=num_frames).astype(int)
    selected_paths = [frame_paths[i] for i in idx]
    frames = [cv2.imread(p) for p in selected_paths]
    return frames


def load_dummy_bboxes(num_frames, num_obj=4, hand=True):
    # Replace this with actual bbox parsing if available
    obj_boxes = torch.zeros((1, num_frames, num_obj, 4))  # [B, T, N_obj, 4]
    hand_boxes = torch.zeros((1, num_frames, 1, 4)) if hand else None
    return obj_boxes, hand_boxes


def predict_segment(rgb_frame_dir, obj_frame_dir, model, cfg):
    frames = load_frames_from_folder(rgb_frame_dir, cfg.DATA.NUM_FRAMES)
    clip = preprocess_frames(frames, cfg)
    inputs = [clip.unsqueeze(0)]  # [1, C, T, H, W]

    # Dummy bounding boxes (replace with real ones if available)
    obj_boxes, hand_boxes = load_dummy_bboxes(cfg.DATA.NUM_FRAMES)
    metadata = {
        "orvit_bboxes": {
            "obj": obj_boxes.to(model.device),
            "hand": hand_boxes.to(model.device),
        }
    }

    inputs = [inp.to(model.device) for inp in inputs]

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
