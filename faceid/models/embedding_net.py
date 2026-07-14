import torch
import torch.nn as nn

from .blocks import ConvBlock


class FaceEmbeddingNet(nn.Module):
    """
    CNN that maps a face image to a 128-dimensional embedding.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.embedding = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a normalized face embedding.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tensor of shape (B, embedding_dim)
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.embedding(x)

        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / (norm + 1e-12)

        return x