dependencies = ["torch"]

from typing import Any

from model import Net, load_model

def cifar10_model(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> Net:
    return load_model(pretrained, progress, **kwargs)