dependencies = ["torch"]

from typing import Any

from model import CIFAR10_Model, load_model

def cifar10_model(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> CIFAR10_Model:
    return load_model(pretrained, progress, **kwargs)