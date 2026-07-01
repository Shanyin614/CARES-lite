"""Model definitions."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

# Kept for backward compatibility with old imports. Do not rely on this for
# tabular models; use get_last_linear_keys instead.
LAST_LAYER_PREFIX = "classifier.2"


def get_last_linear_keys(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, str | None]:
    """Return the weight/bias keys of the final linear layer in a state_dict.

    This works for both SmallCNN (classifier.2.*) and TabularMLP (model.4.*),
    avoiding the Fashion-MNIST-specific LAST_LAYER_PREFIX assumption.
    """
    weight_keys = [
        key
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and torch.is_tensor(tensor) and tensor.ndim == 2
    ]
    if not weight_keys:
        raise RuntimeError("Unable to find a linear layer weight in model state_dict")
    weight_key = weight_keys[-1]
    bias_key = weight_key.replace(".weight", ".bias")
    if bias_key not in state_dict:
        bias_key = None
    return weight_key, bias_key


class SmallCNN(nn.Module):
    """Lightweight CNN for FashionMNIST / CIFAR-10."""

    def __init__(self, input_channels: int = 1, image_size: int = 28, num_classes: int = 10):
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
