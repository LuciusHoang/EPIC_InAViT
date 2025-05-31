import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    """
    CNN + LSTM model for video-based sequence classification.
    Processes video frames with a CNN backbone, then models temporal
    dependencies using an LSTM before classification.
    """

    def __init__(self, cnn_model='resnet18', hidden_size=128, num_layers=2, num_classes=30, dropout=0.3):
        """
        Args:
            cnn_model (str): CNN backbone ('resnet18' recommended).
            hidden_size (int): Hidden size for LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(CNNLSTM, self).__init__()

        # Load pretrained ResNet18 backbone (remove final fc layer)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-1]  # remove final FC layer
        self.cnn = nn.Sequential(*modules)  # Output: (batch_size, 512, 1, 1)
        self.cnn_output_size = 512  # Output features from CNN

        # LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_output_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, seq_length, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        batch_size, seq_length, C, H, W = x.size()

        # Collapse batch and sequence for CNN feature extraction
        x = x.view(batch_size * seq_length, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_length, -1)  # (batch_size, seq_length, cnn_output_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_length, hidden_size)

        # Use final time-step output for classification
        final_features = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        logits = self.fc(final_features)
        return logits
