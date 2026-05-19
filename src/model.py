"""Model definitions."""

import torch.nn as nn

# 最后一层参数名前缀（用于 probe 扰动）
LAST_LAYER_PREFIX = "classifier.2"


class SmallCNN(nn.Module):
    """Lightweight CNN for 28×28 grayscale images (Fashion-MNIST)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # features.0
            nn.ReLU(inplace=True),                         # features.1
            nn.MaxPool2d(2),                               # features.2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # features.3
            nn.ReLU(inplace=True),                         # features.4
            nn.MaxPool2d(2),                               # features.5
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),                    # classifier.0
            nn.ReLU(inplace=True),                         # classifier.1
            nn.Linear(128, num_classes),                    # classifier.2 ← 最后一层
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
