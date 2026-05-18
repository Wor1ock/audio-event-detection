import torch.nn as nn


class HeadFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        return cls._registry[name](**kwargs)
