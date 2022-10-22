import torch
import torchvision

__all__ = ['ToTensor']

class ToTensor(object):
    def __new__(self, n_transform_args: int) -> torchvision.transforms.ToTensor:
        return torchvision.transforms.ToTensor() if n_transform_args < 2 else ToTensorMultiArgs()

class ToTensorMultiArgs(torchvision.transforms.ToTensor):
    # ToTensor preprocess image (0-255 uint -> 0-1 float)
    def __call__(self, *args) -> tuple:
        l = []
        for i, arg in enumerate(args):
            # assume first argument is normalized later
            if i == 0:
                l.append(super(ToTensorMultiArgs, self).__call__(arg))
            else:
                try:
                    l.append(torch.tensor(arg))
                except:
                    l.append(arg)
        return tuple(l)


