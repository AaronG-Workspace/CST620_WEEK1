"""The two architecture families compared in this project: a small CNN and DeiT-tiny."""

import timm
from torch import nn

DEIT_NAME = "deit_tiny_patch16_224"


class SmallCNN(nn.Module):
    """Three conv blocks, global average pooling, and two fully connected layers.

    Input: (N, 3, 224, 224). Each block halves the size: 224 -> 112 -> 56 -> 28.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling: (N, 64, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def create_deit_tiny(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Pretrained DeiT-tiny with a new classification head and a frozen backbone.

    Only the head (192 -> num_classes) has requires_grad=True.
    """
    model = timm.create_model(DEIT_NAME, pretrained=pretrained, num_classes=num_classes)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts."""
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
