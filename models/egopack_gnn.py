# models/egopack_gnn.py

import torch
import torch.nn as nn

class EgoPackGNN(nn.Module):
    def __init__(self, input_dim=512, num_classes=30, add_classifier=True):
        super().__init__()
        # Assume self.gnn exists and returns [B, 512] features
        self.gnn = self._build_dummy_gnn()
        self.add_classifier = add_classifier

        if add_classifier:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, graph, return_embedding=False):
        gnn_emb = self.gnn(graph)  # shape: [B, 512]

        if return_embedding or not self.add_classifier:
            return gnn_emb
        else:
            return self.classifier(gnn_emb)

    def _build_dummy_gnn(self):
        # Placeholder block simulating GNN output
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 512),  # Assume graph input flattens to 1000
            nn.ReLU()
        )
