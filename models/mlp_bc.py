import torch
import torch.nn as nn


class MLPBC(nn.Module):
    """
    Behavior Cloning classifier using an MLP.
    Designed for pose-based input (e.g. 48D pose vector per frame).
    """

    def __init__(self, input_size=48, seq_length=60, hidden_size=256, num_classes=30, dropout=0.3):
        """
        Args:
            input_size (int): Number of features per time step (pose vector size).
            seq_length (int): Number of time steps per sequence.
            hidden_size (int): Number of hidden units in the MLP.
            num_classes (int): Number of classes (tasks).
            dropout (float): Dropout rate.
        """
        super(MLPBC, self).__init__()
        self.flatten = nn.Flatten()  # Flatten entire sequence
        self.mlp = nn.Sequential(
            nn.Linear(input_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.flatten(x)  # shape: (batch_size, seq_length * input_size)
        logits = self.mlp(x)
        return logits
