"""Spectral net model for contrastive learning and classification of spectral data."""

import torch
import torch.nn as nn

class NonLocal1DBlock(nn.Module):
    """
    non-local block for 1D data, from Wang et al. (2018) "Non-local Neural Networks" (https://doi.org/10.48550/arXiv.1711.07971)
    """
    def __init__(self, channels):
        super().__init__()
        self.theta = nn.Conv1d(channels, channels // 2, kernel_size=1, bias=False)
        self.phi = nn.Conv1d(channels, channels // 2, kernel_size=1, bias=False)
        self.g = nn.Conv1d(channels, channels // 2, kernel_size=1, bias=False)
        self.out = nn.Conv1d(channels // 2, channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        # compute query, key, value
        q = self.theta(x).view(B, C // 2, L).permute(0, 2, 1)
        k = self.phi(x).view(B, C // 2, L)
        attn = self.softmax(torch.bmm(q, k))
        v = self.g(x).view(B, C // 2, L).permute(0, 2, 1)
        y = torch.bmm(attn, v).permute(0, 2, 1).view(B, C // 2, L)
        y = self.out(y)
        return x + y


class SpectralNet(nn.Module):
    """
    spectral net model for contrastive learning and classification of spectral data.
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        simclr_hidden_dim=32,
        simclr_z_dim=16,
        encoder_type="conv"
    ):
        """
        Args:
            input_dim: dimension of input spectral data
            num_classes: number of classes for classification
            simclr_hidden_dim: hidden dimension for linear projector
            simclr_z_dim: output dimension for linear projector
            encoder_type: type of encoder to use (conv or linear)
        """

        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "conv":
            print("Using convolutional encoder with non-local attention")
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                NonLocal1DBlock(64),
                
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                
                nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Flatten()
            )
            encoder_out_dim = 8000
        elif encoder_type == "linear":
            print("Using default (linear) encoder")
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU()
            )
            encoder_out_dim = 16
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}.")

        # linear projector for contrastive learning
        self.linear_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, simclr_hidden_dim, bias=False),
            nn.BatchNorm1d(simclr_hidden_dim),
            nn.ReLU(),
            nn.Linear(simclr_hidden_dim, simclr_z_dim, bias=True)
        )
        # linear classifier for supervised finetuning
        self.linear_finetune = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(encoder_out_dim, num_classes),
        )

    def embed(self, x):
        if self.encoder_type == "conv":
            x = x.unsqueeze(1)  # add channel dim
            return self.encoder(x)
        else:
            return self.encoder(x)
        
    def project(self, x):
        x = self.embed(x)
        x = self.linear_projector(x)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.linear_finetune(x)
        return x
