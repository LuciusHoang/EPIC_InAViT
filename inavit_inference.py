import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
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
    clip = torch.stack(processed, dim=1)  # Shape: [C, T, H, W]
    return clip


def sample_frames(video_path, num_frames=16, sampling_rate=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames * sampling_rate).astype(int)[::sampling_rate]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))

    return frames[:num_frames]


def predict_video(video_path, model, cfg):
    frames = sample_frames(
        video_path,
        num_frames=cfg.DATA.NUM_FRAMES,
        sampling_rate=cfg.DATA.SAMPLING_RATE
    )
    clip = preprocess_frames(frames, cfg)

    inputs = clip.unsqueeze(0)  # Shape: [1, C, T, H, W]
    inputs = [inputs]  # Required input format for SlowFast/HOIViT

    # Dummy bounding boxes for HOIViT
    dummy_metadata = {
        "orvit_bboxes": {
            "obj": torch.zeros((1, cfg.DATA.NUM_FRAMES, 4, 4)),
            "hand": torch.zeros((1, cfg.DATA.NUM_FRAMES, 1, 4))
        }
    }

    # Move to CPU or CUDA based on model
    device = next(model.parameters()).device
    inputs = [inp.to(device) for inp in inputs]
    dummy_metadata["orvit_bboxes"]["obj"] = dummy_metadata["orvit_bboxes"]["obj"].to(device)
    dummy_metadata["orvit_bboxes"]["hand"] = dummy_metadata["orvit_bboxes"]["hand"].to(device)

    with torch.no_grad():
        logits = model(inputs, dummy_metadata)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return pred_idx.item(), conf.item()


def build_and_load_model(cfg, pretrained_path="checkpoints/checkpoint_epoch_00081.pyth"):
    model = build_model(cfg)

    # ✅ Inject required default_cfg keys for load_pretrained()
    model.default_cfg = {
        "first_conv": "patch_embed.proj",   # HOIViT input stem
        "classifier": "head",               # default FC layer name
        "url": ""                           # not used since we load from file
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
    return model
