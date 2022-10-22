from pathlib import Path

import torch

from torchfetch.custom.metaclass import Singleton

from .private import PrivateModelLoader
from .public import is_public_architecture, instantiate_public_network


KEY_PATH_CHECKPOINT = "path_checkpoint"


__all__ = ['ModelLoader',
           'get_pretrained_network', 'instantiate_network', 'get_checkpoint', 'get_model_state_dict',
           'get_optimizer_state_dict', 'get_preprocess_name_used_for_train']


class ModelLoader(object, metaclass=Singleton):

    def instantiate_network(self, target: str, **kwargs) -> torch.nn:
        if is_public_architecture(target):
            return instantiate_public_network(target, **kwargs)
        else:
            return PrivateModelLoader().instantiate_network(Path(target))


get_pretrained_network = PrivateModelLoader().load
instantiate_network = ModelLoader().instantiate_network
get_checkpoint = PrivateModelLoader().get_checkpoint
get_model_state_dict = PrivateModelLoader().get_model_state_dict
get_hyperparam = PrivateModelLoader().get_hyperparam
get_optimizer_state_dict = PrivateModelLoader().get_optimizer_state_dict
get_preprocess_name_used_for_train = PrivateModelLoader().get_preprocess_name_used_for_train
