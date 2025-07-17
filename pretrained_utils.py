import math
import torch
import torch.nn.functional as F
from timm.models.helpers import load_pretrained as model_zoo

def _conv_filter(state_dict):
    return state_dict  # passthrough by default

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=_conv_filter,
                    img_size=224, num_frames=8, num_patches=196, attention_type='divided_space_time',
                    pretrained_model="", strict=True):

    if cfg is None:
        cfg = getattr(model, 'default_cfg', {})
    if cfg is None or not cfg.get('url', ''):
        if not pretrained_model:
            print("⚠️ Pretrained model URL is invalid or not provided.")
            return

    # Load checkpoint
    if pretrained_model:
        checkpoint = torch.load(pretrained_model, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    else:
        state_dict = model_zoo(cfg['url'], progress=False, map_location='cpu')

    # Optional filtering
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    # Adjust first conv layer
    conv1_name = cfg.get('first_conv', 'patch_embed.proj')
    if in_chans == 1:
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_weight = state_dict[conv1_name + '.weight']
        if conv1_weight.shape[1] != 3:
            print(f"⚠️ Skipping incompatible conv layer: {conv1_name}")
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            repeat = math.ceil(in_chans / 3)
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            state_dict[conv1_name + '.weight'] = conv1_weight

    # Handle classifier layer
    classifier_name = cfg.get('classifier', 'head')
    if classifier_name + '.weight' in state_dict:
        current_classes = state_dict[classifier_name + '.weight'].shape[0]
        if current_classes != num_classes:
            print(f"🔁 Removing pretrained classifier head: expected {num_classes}, found {current_classes}")
            state_dict.pop(classifier_name + '.weight', None)
            state_dict.pop(classifier_name + '.bias', None)
            strict = False
    else:
        print(f"⚠️ Classifier weights '{classifier_name}.weight' not found in checkpoint — skipping removal.")

    # Resize positional embeddings
    if 'pos_embed' in state_dict and num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_token = pos_embed[:, :1, :]
        spatial_tokens = pos_embed[:, 1:, :].permute(0, 2, 1)
        spatial_tokens = F.interpolate(spatial_tokens, size=num_patches, mode='nearest')
        new_pos_embed = torch.cat([cls_token, spatial_tokens.permute(0, 2, 1)], dim=1)
        state_dict['pos_embed'] = new_pos_embed

    # Resize time embeddings
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].permute(0, 2, 1)
        time_embed = F.interpolate(time_embed, size=num_frames, mode='nearest')
        state_dict['time_embed'] = time_embed.permute(0, 2, 1)

    # Resize patch embedding
    if 'patch_embed.proj.weight' in state_dict:
        target_shape = model.state_dict()['patch_embed.proj.weight'].shape[-3:]
        kernel = state_dict['patch_embed.proj.weight']
        kernel = F.interpolate(kernel, size=target_shape[1:], mode='nearest')
        kernel = kernel.unsqueeze(2).expand(-1, -1, target_shape[0], -1, -1)
        state_dict['patch_embed.proj.weight'] = kernel

    # Rename positional tokens
    if 'pos_embed' in state_dict:
        state_dict['pos_embed_class'] = state_dict['pos_embed'][:, :1]
        state_dict['pos_embed_spatial'] = state_dict['pos_embed'][:, 1:]
        del state_dict['pos_embed']

    # Load the state dict
    model.load_state_dict(state_dict, strict=strict)

    # Ensure classifier head matches target num_classes
    classifier = getattr(model, classifier_name, None)
    if isinstance(classifier, torch.nn.Linear):
        out_features = classifier.out_features
        if out_features != num_classes:
            print(f"🔁 Resetting classifier head to {num_classes} classes.")
            in_features = classifier.in_features
            new_fc = torch.nn.Linear(in_features, num_classes)
            setattr(model, classifier_name, new_fc)
