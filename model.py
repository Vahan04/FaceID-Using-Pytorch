import numpy as np
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class FaceEmbeddingNet(nn.Module):
    """
    CNN that maps a face image to a 128-dimensional embedding.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.adaptive_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 128)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool2d(x)
        x = self.flatten(x)
        x = self.linear(x)
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / (norm + 1e-12)
        return x
        
model = FaceEmbeddingNet()

x = torch.rand(1, 3, 160, 160)

y = model(x)

print(x.shape)
print(y.shape)
embedding_norm = torch.norm(y, p=2, dim=1, keepdim=True)
print(embedding_norm)