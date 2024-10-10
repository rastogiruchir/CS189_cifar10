from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

__all__ = ["CIFAR10_Model", "load_model"]

model_urls = {
    "model": "https://github.com/rastogiruchir/CS189_cifar10/blob/main/classifier.pth"
}

class CIFAR10_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(256 * 8 * 8, 256)
        self.bn_l1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.bn2(F.relu(self.conv2(out)))
        out = self.pool1(out)
        out = self.bn3(F.relu(self.conv3(out)))
        out = self.bn4(F.relu(self.conv4(out)))
        out = self.pool2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.bn_l1(F.relu(self.linear1(out)))
        out = self.linear2(out)
        return out


def load_model(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> CIFAR10_Model:
    model = CIFAR10_Model()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["model"], progress=progress)
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("Model not available")
    return model
