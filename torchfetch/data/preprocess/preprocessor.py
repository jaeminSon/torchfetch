import torch
import torchvision
from pathlib import PurePath

from torchfetch.descriptor import PreprocessDescriptor
from torchfetch.custom.utils import read_json

from torchfetch.custom.typing import Preprocess

__all__ = ['Preprocessor']


class Preprocessor(object):

    def __new__(self, preprocess: Preprocess, n_transform_args: int = None) -> torch.nn.Module:

        if preprocess is None:
            if n_transform_args < 2:
                return torchvision.transforms.Normalize(mean=[0], std=[1])
            else:
                return MultiArgsNormalize(mean=[0], std=[1])
        else:
            info = Preprocessor.read_preprocess(preprocess)
            if (PreprocessDescriptor.KEY_MEAN in info and PreprocessDescriptor.KEY_STD in info):
                mean = info[PreprocessDescriptor.KEY_MEAN]
                std = info[PreprocessDescriptor.KEY_STD]
                if n_transform_args < 2:
                    return torchvision.transforms.Normalize(mean=mean, std=std)
                else:
                    return MultiArgsNormalize(mean=mean, std=std)
            else:
                raise NotImplementedError("Unknown preprocess: {}. Currently support (x-'mean')/'std'.".format(info))

    @staticmethod
    def read_preprocess(preprocess: Preprocess):
        if type(preprocess) == str:
            return read_json(preprocess)
        elif isinstance(preprocess, PurePath):
            return read_json(preprocess)
        elif Preprocessor.is_preprocess(preprocess):
            return preprocess
        else:
            raise ValueError("Augment should be str or Path or dictionary with 'mean' and 'std' keys.")

    @staticmethod
    def is_preprocess(info: dict):
        return PreprocessDescriptor.KEY_MEAN in info and PreprocessDescriptor.KEY_STD in info


class MultiArgsNormalize(torchvision.transforms.Normalize):
    def forward(self, *args):
        # normalize first argument only
        return (super().forward(args[0]),) + args[1:]
