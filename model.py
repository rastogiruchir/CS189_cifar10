from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    "model": "https://github.com/rastogiruchir/CS189_cifar10/blob/main/cifar10_classifier.pt"
}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(256 * 16 * 16, 256)
        self.bn_l1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.bn3(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def load_model(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> Net:
    model = Net()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["model"], progress=progress)
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("Model not available")
    return model
