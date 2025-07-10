# models/egom2p_encoder.py

import torch
import torch.nn as nn

class EgoM2PEncoder(nn.Module):
    def __init__(self, input_channels=3, pose_dim=21*3, hidden_dim=512, num_classes=30, add_classifier=True):
        super().__init__()
        self.rgb_conv = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.pose_fc = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.cls_token = nn.Linear(32 + 128, hidden_dim)
        self.add_classifier = add_classifier

        if add_classifier:
            self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, rgb, pose, return_embedding=False):
        rgb = rgb.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] → [B, C, T, H, W]
        rgb_feat = self.rgb_conv(rgb).view(rgb.size(0), -1)

        B, T, J, D = pose.shape
        pose = pose.view(B, T, J * D).mean(dim=1)
        pose_feat = self.pose_fc(pose)

        cls_emb = self.cls_token(torch.cat([rgb_feat, pose_feat], dim=1))  # [B, 512]

        if return_embedding or not self.add_classifier:
            return cls_emb
        else:
            return self.classifier(cls_emb)  # [B, num_classes]
