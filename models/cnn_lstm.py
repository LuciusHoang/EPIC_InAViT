import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """
    Adapted CNNLSTM model for pose-based sequence classification.
    Instead of a CNN backbone, this version uses a Linear layer to project
    input pose features, followed by an LSTM and classifier.
    """

    def __init__(self, input_dim=36, hidden_size=128, num_layers=2,
                 num_classes=30, dropout=0.3):
        """
        Args:
            input_dim (int): Number of features per time step (pose vector size).
            hidden_size (int): Hidden size for LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(CNNLSTM, self).__init__()

        # Input projection layer (instead of CNN)
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Step 1: Project input features
        x = self.input_proj(x)  # (batch_size, seq_length, hidden_size)

        # Step 2: Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)

        # Step 3: Use final time-step output
        final_features = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Step 4: Classify
        logits = self.classifier(final_features)  # (batch_size, num_classes)
        return logits
