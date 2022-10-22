import torch
import torchvision

__all__ = ['MultiArgsCompose']


class MultiArgsCompose(torchvision.transforms.Compose):
    def __call__(self, *data) -> torch.Tensor:
        for t in self.transforms:
            data = t(*data)
        return data

