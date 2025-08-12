import math
import torch
import torch.nn.functional as F
from timm.models.helpers import load_pretrained as model_zoo

def _conv_filter(state_dict):
    # passthrough by default; hook to modify incoming keys/weights if needed
    return state_dict

def _extract_state_dict(ckpt):
    """
    Try common places for the model weights inside a checkpoint blob.
    """
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # Some checkpoints are just the raw state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict).")

def load_pretrained(
    model,
    cfg=None,
    num_classes=1000,
    in_chans=3,
    filter_fn=_conv_filter,
    img_size=224,
    num_frames=8,
    num_patches=196,
    attention_type="divided_space_time",
    pretrained_model="",
    strict=True,
):
    """
    Load pretrained weights in a device-agnostic way (CPU-safe),
    optionally adapting first conv, classifier, and positional/time embeddings.
    """

    # Resolve default cfg
    if cfg is None:
        cfg = getattr(model, "default_cfg", {}) or {}

    url = cfg.get("url", "")

    # ---- Load checkpoint to CPU (works for CPU/MPS/CUDA later) ----
    state_dict = None
    if pretrained_model:
        try:
            ckpt = torch.load(pretrained_model, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            print(f"🔹 Loaded checkpoint from: {pretrained_model}")
        except Exception as e:
            print(f"⚠️ Failed to load local checkpoint '{pretrained_model}': {e}")
    if state_dict is None and url:
        try:
            state_dict = model_zoo(url, progress=False, map_location="cpu")
            print(f"🔹 Loaded checkpoint from URL (timm): {url}")
        except Exception as e:
            print(f"⚠️ Failed to load URL checkpoint: {e}")

    if state_dict is None:
        print("⚠️ No pretrained weights could be loaded; continuing with random init.")
        return

    # ---- Optional filtering hook ----
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    # ---- Adapt first conv / patch embed for in_chans ----
    conv1_name = cfg.get("first_conv", "patch_embed.proj")
    conv_w_key = conv1_name + ".weight"
    if conv_w_key in state_dict:
        conv_w = state_dict[conv_w_key]
        # Handle grayscale or >3 channels
        if in_chans == 1 and conv_w.dim() >= 3 and conv_w.shape[1] != 1:
            # Sum RGB to 1 channel
            state_dict[conv_w_key] = conv_w.sum(dim=1, keepdim=True)
        elif in_chans != 3 and conv_w.dim() >= 3:
            if conv_w.shape[1] == 3:
                repeat = math.ceil(in_chans / 3)
                conv_w = conv_w.repeat(1, repeat, *([1] * (conv_w.dim() - 2)))[:, :in_chans, ...]
                conv_w *= (3.0 / float(in_chans))
                state_dict[conv_w_key] = conv_w
            else:
                # Channel mismatch that we can't trivially fix: drop and load non-strict
                print(f"⚠️ Incompatible input channels for '{conv1_name}'; skipping that weight.")
                state_dict.pop(conv_w_key, None)
                strict = False

    # ---- Classifier head handling ----
    classifier_name = cfg.get("classifier", "head")
    head_w_key = classifier_name + ".weight"
    head_b_key = classifier_name + ".bias"
    if head_w_key in state_dict:
        current_classes = state_dict[head_w_key].shape[0]
        if current_classes != num_classes:
            print(f"🔁 Removing pretrained classifier head: expected {num_classes}, found {current_classes}")
            state_dict.pop(head_w_key, None)
            state_dict.pop(head_b_key, None)
            strict = False
    else:
        # Not all checkpoints include the head (common for backbone-only)
        pass

    # ---- Positional embeddings (spatial tokens) ----
    # Expecting shape [B, 1+N, C]; if N (num_patches) differs, resize spatial tokens only.
    pe_key = "pos_embed"
    if pe_key in state_dict:
        pe = state_dict[pe_key]
        if pe.ndim == 3 and pe.size(1) != (num_patches + 1):
            cls_tok = pe[:, :1, :]                  # [B, 1, C]
            spa = pe[:, 1:, :]                      # [B, N_old, C]
            # Interpolate along token dimension (treat as 1D “image”)
            spa = spa.permute(0, 2, 1)              # [B, C, N_old]
            spa = F.interpolate(spa, size=num_patches, mode="nearest")
            spa = spa.permute(0, 2, 1)              # [B, N_new, C]
            state_dict[pe_key] = torch.cat([cls_tok, spa], dim=1)

    # ---- Temporal embeddings ----
    # Expecting shape [B, T, C]; if T differs, resize along T.
    te_key = "time_embed"
    if te_key in state_dict:
        te = state_dict[te_key]
        if te.ndim == 3 and te.size(1) != num_frames:
            te = te.permute(0, 2, 1)                # [B, C, T_old]
            te = F.interpolate(te, size=num_frames, mode="nearest")
            te = te.permute(0, 2, 1)                # [B, T_new, C]
            state_dict[te_key] = te

    # ---- DO NOT reshape kernel sizes for patch_embed here ----
    # If kernel/HW mismatch occurs, rely on strict=False and layer re-init.

    # ---- Load weights ----
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"ℹ️ Missing keys: {len(missing)} (showing up to 10)\n  " + "\n  ".join(missing[:10]))
    if unexpected:
        print(f"ℹ️ Unexpected keys: {len(unexpected)} (showing up to 10)\n  " + "\n  ".join(unexpected[:10]))

    # ---- Ensure classifier matches num_classes ----
    head = getattr(model, classifier_name, None)
    if isinstance(head, torch.nn.Linear):
        if head.out_features != num_classes:
            print(f"🔁 Resetting classifier head to {num_classes} classes.")
            in_features = head.in_features
            setattr(model, classifier_name, torch.nn.Linear(in_features, num_classes))
    else:
        # Some models wrap the head; if needed, add handling here.
        pass
