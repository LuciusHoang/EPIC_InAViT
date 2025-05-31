import torch
import torch.nn as nn


class PoseTransformer(nn.Module):
    """
    Transformer Encoder for sequence classification using pose-based inputs.
    """

    def __init__(self, input_size=48, seq_length=60, d_model=128, nhead=4, num_layers=4, num_classes=30, dropout=0.3):
        """
        Args:
            input_size (int): Number of features per time step (pose vector size).
            seq_length (int): Number of time steps per sequence.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(PoseTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=4 * d_model,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.input_proj(x)  # (batch_size, seq_length, d_model)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, d_model) for transformer

        output = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        output = output[-1, :, :]  # Use the last time step output for classification

        logits = self.classifier(output)
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to inject position information into the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
