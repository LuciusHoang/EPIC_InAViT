import torch
import torch.nn as nn

class MLPBC(nn.Module):
    """
    Behavior Cloning classifier using a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, input_dim=36, seq_length=60, hidden_size=256,
                 num_classes=30, dropout=0.3):
        super(MLPBC, self).__init__()

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * seq_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits
