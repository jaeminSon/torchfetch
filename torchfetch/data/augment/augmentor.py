from pathlib import Path, PurePath

from .image import ImageMaskAugmentor, SingleImageAugmentor, ImageBboxAugmentor

from torchfetch.custom.utils import is_albumentations
from torchfetch.descriptor import InputTypeDescriptor, AugmentDescriptor, TaskDescriptor

from torchfetch.custom.typing import Augment

__all__ = ['Augmentor']


class Augmentor(object):
    def __new__(self, augment: Augment, task: str = None) -> object:

        if augment is None:
            if task == TaskDescriptor.NAME_DETECTION:
                return MultiNullAugmentor(n_aug_args=2) # image, mask
            elif task == TaskDescriptor.NAME_SEGMENTATION:
                return MultiNullAugmentor(n_aug_args=2) # image, label
            else:
                return NullAugmentor()
        elif type(augment) == str:
            augment = Path(augment)
        elif isinstance(augment, PurePath) or is_albumentations(augment):
            pass
        else:
            raise ValueError("Augment should be str or Path or albumentations(Compose object).")

        if AugmentDescriptor().infer_input_type_from_augment(augment) == InputTypeDescriptor.NAME_IMAGE:
            if task == TaskDescriptor.NAME_DETECTION:
                return ImageBboxAugmentor(augment)
            elif task == TaskDescriptor.NAME_SEGMENTATION:
                return ImageMaskAugmentor(augment)
            else:
                return SingleImageAugmentor(augment)
        else:
            raise NotImplementedError()


class NullAugmentor(object):
    def __call__(self, x):
        return x

    @property
    def n_aug_args(self):
        return 1


class MultiNullAugmentor(object):
    
    def __init__(self, n_aug_args):
        self.n_aug_args = n_aug_args
    
    def __call__(self, *args):
        return args
