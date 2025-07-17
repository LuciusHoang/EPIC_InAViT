import torch
import torch.nn as nn
import os

# ✅ Updated import paths using InAViT.slowfast
from InAViT.slowfast.config.defaults import get_cfg
from InAViT.slowfast.models.build import build_model


def load_model(checkpoint_path, cfg_path="EK_INAVIT_MF_ant.yaml"):
    """
    Load InAViT model from pretrained checkpoint and config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.TRAIN.ENABLE = False   # Inference only
    cfg.TEST.ENABLE = True
    cfg.freeze()

    model = build_model(cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    return model


def predict_video(model, frames):
    """
    Run inference on a single video clip (frames: [B, C, T, H, W]).
    """
    model.eval()
    device = next(model.parameters()).device
    frames = frames.to(device)

    with torch.no_grad():
        preds = model(frames)  # Output: [B, num_classes]
    return preds
