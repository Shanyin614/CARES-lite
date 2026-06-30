"""Model definitions."""

import torch.nn as nn

# 最后一层参数名前缀（用于 probe 扰动）
LAST_LAYER_PREFIX = "classifier.2"


class SmallCNN(nn.Module):
    """Lightweight CNN for FashionMNIST / CIFAR-10."""

    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 28,
        num_classes: int = 10,
    ):
        super().__init__()

        if image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {image_size}")

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        feat_size = image_size // 4

        self.classifier = nn.Sequential(
            nn.Linear(32 * feat_size * feat_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class TabularMLP(nn.Module):
    """Lightweight MLP for tabular intrusion detection datasets."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
