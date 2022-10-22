from pathlib import PurePath

import albumentations as A

from torchfetch.custom.utils import read_json, read_yaml, is_albumentations, loadable_albumentations_file
from torchfetch.custom.typing import Augment
from torchfetch.custom.metaclass import Singleton

from .inputtype import InputTypeDescriptor

__all__ = ['AugmentDescriptor']


class AugmentDescriptor(object, metaclass=Singleton):

    NAME_AUGMENT_DIR = "augment"

    NAME_ALBUMENTATIONS = "albumentations"

    LIBRARY_IMAGES = [NAME_ALBUMENTATIONS]

    def infer_input_type_from_augment(self, augment: Augment) -> str:
        aug_lib = self.infer_augment_library(augment)
        return self.infer_input_type_from_aug_lib(aug_lib)

    def infer_augment_library(self, augment: Augment) -> str:
        if is_albumentations(augment) or loadable_albumentations_file(augment):
            return AugmentDescriptor.NAME_ALBUMENTATIONS
        else:
            raise NotImplementedError

    def infer_input_type_from_aug_lib(self, augment_library: str) -> str:
        if augment_library in AugmentDescriptor.LIBRARY_IMAGES:
            return InputTypeDescriptor.NAME_IMAGE

    def is_detection_task(self, img_aug: Augment) -> bool:
        if is_albumentations(img_aug):
            return img_aug._to_dict()["bbox_params"] is not None
        else:
            try:
                return A.load(img_aug)._to_dict()["bbox_params"] is not None
            except:
                try:
                    return A.load(img_aug, data_format="yaml")._to_dict()["bbox_params"] is not None
                except:
                    return False
