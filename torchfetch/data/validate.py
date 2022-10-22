from pathlib import Path

from .preprocess import Preprocessor
from .augment import Augmentor

from torchfetch.custom.typing import FilePath
from torchfetch.custom.decorator import return_true_if_pass_else_false

__all__ = ['Validator']


class Validator(object):

    @return_true_if_pass_else_false
    def is_valid_augment(self, filepath: FilePath) -> bool:
        Augmentor(augment=Path(filepath))

    @return_true_if_pass_else_false
    def is_valid_preprocess(self, filepath: FilePath) -> bool:
        Preprocessor(preprocess=Path(filepath), n_transform_args=1)

is_valid_augment = Validator().is_valid_augment
is_valid_preprocess = Validator().is_valid_preprocess