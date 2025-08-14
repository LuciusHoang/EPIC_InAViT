import math
import torch
import torch.nn.functional as F
from timm.models.helpers import load_pretrained as model_zoo


# ------------------------------------------------------------
# Optional hook to massage incoming checkpoints (rename keys, etc.)
# Keep as passthrough unless you need to adapt third‑party weights.
# ------------------------------------------------------------
def _conv_filter(state_dict: dict) -> dict:
    return state_dict


# ------------------------------------------------------------
# Robustly extract a state_dict from various checkpoint formats
# ------------------------------------------------------------
def _extract_state_dict(ckpt):
    """
    Try common places for the model weights inside a checkpoint blob.
    Returns a state_dict (dict[str, Tensor]).
    """
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state", "net", "weights"]:
            v = ckpt.get(k, None)
            if isinstance(v, dict):
                return v
        # Some checkpoints are just the raw state_dict
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected dict).")


# ------------------------------------------------------------
# Positional & Temporal embedding resizing helpers
# ------------------------------------------------------------
def _resize_pos_embed_1d(pe: torch.Tensor, num_patches_new: int) -> torch.Tensor:
    """
    Resize ViT-style position embedding (B, 1+N, C) along token dimension.
    Keeps the [CLS] token, resizes the N spatial tokens with nearest interpolation.
    """
    if pe.ndim != 3 or pe.size(1) < 2:
        return pe
    cls_tok = pe[:, :1, :]   # (B,1,C)
    spa = pe[:, 1:, :]       # (B,N_old,C)
    if spa.size(1) == num_patches_new:
        return pe
    spa = spa.permute(0, 2, 1)                # (B,C,N_old)
    spa = F.interpolate(spa, size=num_patches_new, mode="nearest")
    spa = spa.permute(0, 2, 1)                # (B,N_new,C)
    return torch.cat([cls_tok, spa], dim=1)   # (B,1+N_new,C)


def _resize_time_embed(te: torch.Tensor, T_new: int) -> torch.Tensor:
    """
    Resize temporal embedding (B, T, C) to T_new with nearest interpolation.
    """
    if te.ndim != 3 or te.size(1) == T_new:
        return te
    te = te.permute(0, 2, 1)                  # (B,C,T_old)
    te = F.interpolate(te, size=T_new, mode="nearest")
    te = te.permute(0, 2, 1)                  # (B,T_new,C)
    return te


# ------------------------------------------------------------
# Main loader
# ------------------------------------------------------------
def load_pretrained(
    model,
    cfg: dict = None,
    *,
    # For InAViT we typically **keep** checkpoint classifier heads as-is.
    keep_classifier: bool = True,
    # If you *must* override, you can pass either a single int or a mapping.
    # Example mapping: {"head_verb": 97, "head_noun": 300, "head_action": 3805}
    num_classes=None,
    in_chans: int = 3,
    filter_fn=_conv_filter,
    img_size: int = 224,
    num_frames: int = 16,
    num_patches: int = 196,
    attention_type: str = "divided_space_time",  # unused but preserved for API compatibility
    pretrained_model: str = "",
    strict: bool = True,
):
    """
    CPU-safe, shape-tolerant loader with optional embedding resizing.

    Defaults are chosen to **preserve** classifier heads from the checkpoint
    (keep_classifier=True), which is what you want for InAViT so you don’t
    accidentally remap 97/300/3805(6) to a subset.

    Parameters
    ----------
    model : torch.nn.Module
    cfg   : dict-like 'default_cfg' with optional keys:
            - 'url': remote weights (timm-style)
            - 'first_conv': name of patch/embed conv (e.g., 'patch_embed.proj')
            - 'classifier': string or list of classifier module names
                            (e.g., 'head' or ['head_verb','head_noun','head_action'])
    keep_classifier : bool
        If True, do **not** alter/remove classifier weights even when shapes differ.
        If False, will reset classifier(s) to `num_classes` after loading.
    num_classes : int or dict[str,int] or None
        Target class count(s) used only when keep_classifier=False.
    in_chans : int
        Number of input channels (adapts first conv if needed).
    filter_fn : callable
        Optional function that mutates the incoming checkpoint state_dict.
    img_size, num_frames, num_patches :
        Used to resize positional/temporal embeddings when shapes differ.
    pretrained_model : str
        Local checkpoint path. If empty and cfg['url'] provided, will try timm’s loader.
    strict : bool
        Passed to model.load_state_dict. Will be relaxed internally if we drop/resize keys.
    """

    # Resolve default cfg
    if cfg is None:
        cfg = getattr(model, "default_cfg", {}) or {}

    url = cfg.get("url", "")
    first_conv = cfg.get("first_conv", "patch_embed.proj")

    # Accept classifier as str or list
    classifier = cfg.get("classifier", "head")
    if isinstance(classifier, str):
        classifier_list = [classifier]
    elif isinstance(classifier, (list, tuple)):
        classifier_list = list(classifier)
    else:
        classifier_list = ["head"]

    # ---- Load checkpoint to CPU (works on CPU/MPS/CUDA later) ----
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
        try:
            state_dict = filter_fn(state_dict)
        except Exception as e:
            print(f"⚠️ filter_fn raised an exception; continuing with unfiltered weights: {e}")

    # ---- Adapt first conv / patch embed for in_chans ----
    conv_w_key = first_conv + ".weight"
    if conv_w_key in state_dict:
        conv_w = state_dict[conv_w_key]
        # Handle grayscale or >3 channels by simple replication / reduction
        if in_chans == 1 and conv_w.dim() >= 3 and conv_w.shape[1] != 1:
            state_dict[conv_w_key] = conv_w.sum(dim=1, keepdim=True)
        elif in_chans != 3 and conv_w.dim() >= 3:
            if conv_w.shape[1] == 3:
                repeat = math.ceil(in_chans / 3)
                conv_w = conv_w.repeat(1, repeat, *([1] * (conv_w.dim() - 2)))[:, :in_chans, ...]
                conv_w *= (3.0 / float(in_chans))
                state_dict[conv_w_key] = conv_w
            else:
                print(f"⚠️ Incompatible input channels for '{first_conv}'; dropping that weight.")
                state_dict.pop(conv_w_key, None)
                strict = False

    # ---- Positional embeddings ----
    # Expected shape: [B, 1+N, C]. If N differs from num_patches, resize spatial tokens only.
    pe_key = "pos_embed"
    if pe_key in state_dict:
        try:
            state_dict[pe_key] = _resize_pos_embed_1d(state_dict[pe_key], num_patches_new=num_patches)
        except Exception as e:
            print(f"⚠️ Could not resize pos_embed; keeping original. Reason: {e}")
            # Keep strict as-is; pos_embed mismatch might be tolerated by the model.

    # ---- Temporal embeddings ----
    # Expected shape: [B, T, C]. If T differs, resize along T.
    te_key = "time_embed"
    if te_key in state_dict:
        try:
            state_dict[te_key] = _resize_time_embed(state_dict[te_key], T_new=num_frames)
        except Exception as e:
            print(f"⚠️ Could not resize time_embed; keeping original. Reason: {e}")

    # ---- Classifier heads handling ----
    # For InAViT, we *prefer to keep* checkpoint heads intact (97/300/3805 or 3806).
    if keep_classifier:
        # Do NOT remove/resize any classifier weights. Just try to load as-is.
        pass
    else:
        # Optionally reset classifiers to user-specified num_classes
        # Accept a single int or a dict mapping head names to sizes.
        numcls_map = {}
        if isinstance(num_classes, int):
            for head_name in classifier_list:
                numcls_map[head_name] = num_classes
        elif isinstance(num_classes, dict):
            numcls_map = num_classes.copy()
        else:
            # If None or unsupported type, do nothing (keep checkpoint heads).
            numcls_map = {}

        for head_name in classifier_list:
            w_key = f"{head_name}.weight"
            b_key = f"{head_name}.bias"
            if head_name in numcls_map:
                # Drop checkpoint head so model will keep its own randomly initialized head size
                if w_key in state_dict or b_key in state_dict:
                    print(f"🔁 Dropping checkpoint head '{head_name}' to reset to {numcls_map[head_name]} classes.")
                    state_dict.pop(w_key, None)
                    state_dict.pop(b_key, None)
                    strict = False

    # ---- Load state dict ----
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"ℹ️ Missing keys: {len(missing)} (showing up to 12)\n  " + "\n  ".join(missing[:12]))
    if unexpected:
        print(f"ℹ️ Unexpected keys: {len(unexpected)} (showing up to 12)\n  " + "\n  ".join(unexpected[:12]))

    # ---- Ensure classifier heads exist with desired sizes if keep_classifier=False ----
    if not keep_classifier and num_classes is not None:
        # Reset heads on the live model to requested sizes
        if isinstance(num_classes, int):
            for head_name in classifier_list:
                _reset_linear_head(model, head_name, num_classes)
        elif isinstance(num_classes, dict):
            for head_name, nc in num_classes.items():
                _reset_linear_head(model, head_name, nc)


# ------------------------------------------------------------
# Helper to reset a torch.nn.Linear head safely
# ------------------------------------------------------------
def _reset_linear_head(model, head_attr: str, out_features: int):
    head = getattr(model, head_attr, None)
    if isinstance(head, torch.nn.Linear):
        if head.out_features != out_features:
            print(f"🔁 Resetting classifier '{head_attr}' to {out_features} classes.")
            in_features = head.in_features
            setattr(model, head_attr, torch.nn.Linear(in_features, out_features))
    else:
        # Some models wrap the head (e.g., nn.Sequential). Extend as needed.
        pass
