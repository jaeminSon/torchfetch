import inspect
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import torch

from torchfetch.custom.typing import FilePath
from torchfetch.custom.metaclass import Singleton
from torchfetch.custom.utils import get_arguments, read_json, relative2absolute
from torchfetch.descriptor import RecipeDescriptor
from torchvision import models

__all__ = ['PrivateModelLoader']


class PrivateModelLoader(object, metaclass=Singleton):

    def load(self, path_recipe: FilePath) -> torch.nn:
        path_recipe = Path(path_recipe)
        network = self.instantiate_network(path_recipe)
        network = self.load_network_weight_from_recipe_name(
            network, path_recipe)
        return network

    def instantiate_network(self, path_recipe: Path) -> torch.nn:
        return self.instantiate_network_from_recipe_path(path_recipe)

    def instantiate_network_from_recipe_path(self, path_recipe: FilePath) -> torch.nn:
        path_recipe = Path(path_recipe)
        assert path_recipe.exists(), "[Load failed] No recipe file exists."

        str_path_arch = RecipeDescriptor().get_architecture_filepath_from_recipe_file(path_recipe)
        arch_classname = RecipeDescriptor().get_architecture_classname_from_recipe_file(path_recipe)
        if str_path_arch is None and self.is_torchvision_architecture(arch_classname):
            network = self.instantiate_network_from_torch_library(
                path_recipe, arch_classname)
        else:
            path_arch = Path(str_path_arch)
            assert path_arch.exists(
            ), "[Load failed] Architecture not specified. Only checkpoints are loadable."
            network = self.instantiate_network_from_files(
                path_recipe, path_arch)
        return network

    def instantiate_network_from_torch_library(self, path_recipe: Path, arch_name: str) -> torch.nn:
        network_class = getattr(models, arch_name)
        kwargs = RecipeDescriptor().get_architecture_args_from_recipe_file(
            path_recipe)
        network = network_class(**kwargs)
        return network

    def load_network_weight_from_recipe_name(self, network: torch.nn, path_recipe: Path) -> torch.nn:
        return self.load_network_weight_from_recipe_path(network, path_recipe)

    def load_network_weight_from_recipe_path(self, network: torch.nn, path_recipe: Path) -> torch.nn:
        checkpoint_filename = RecipeDescriptor().get_checkpoint_filepath_from_recipe_file(path_recipe)
        path_checkpoint = RecipeDescriptor().get_checkpoint_filepath_from_recipe_file(path_recipe)
        return self.load_network_weight_from_files(network, path_recipe, path_checkpoint)

    def load_network_weight_from_files(self, network: torch.nn, path_recipe: Path, path_checkpoint: Path) -> torch.nn:
        key_weight = RecipeDescriptor().get_checkpoint_keyweight_from_recipe_file(
            path_recipe)
        network = self.load_network_weight(
            network, path_checkpoint, key_weight)
        return network

    def get_model_state_dict(self, path_recipe: FilePath) -> OrderedDict:
        checkpoint = self.get_checkpoint(path_recipe)
        key_weight = RecipeDescriptor().get_checkpoint_keyweight_from_recipe_file(
            path_recipe)
        return checkpoint[key_weight]

    def get_optimizer_state_dict(self, path_recipe: FilePath) -> OrderedDict:
        checkpoint = self.get_checkpoint(path_recipe)
        key_optimizer = RecipeDescriptor().get_checkpoint_keyoptimizer_from_recipe_file(
            path_recipe)
        return checkpoint[key_optimizer]

    def get_checkpoint(self, path_recipe: FilePath) -> dict:
        path_checkpoint = RecipeDescriptor().get_checkpoint_filepath_from_recipe_file(path_recipe)
        checkpoint = self.load_checkpoint_file(path_checkpoint)
        return checkpoint

    def get_hyperparam(self, path_hyperparam: FilePath) -> dict:
        return read_json(path_hyperparam)

    def instantiate_network_from_files(self, path_recipe: Path, path_arch: Path) -> torch.nn:
        classname = RecipeDescriptor().get_architecture_classname_from_recipe_file(
            path_recipe)
        network_class = self.get_classobject_from_file(path_arch, classname)
        kwargs = RecipeDescriptor().get_architecture_args_from_recipe_file(
            path_recipe)
        network = network_class(**kwargs)
        return network

    def get_preprocess_name_used_for_train(self, path_recipe: FilePath) -> str:
        recipe = read_json(path_recipe)
        return recipe[RecipeDescriptor.KEY_PREPROCESS]

    def load_checkpoint_file(self, path_checkpoint: Path) -> dict:
        return torch.load(path_checkpoint)

    def load_network_weight(self, network: torch.nn, path_weight: Path, key_weight: str) -> torch.nn:
        checkpoint = self.load_checkpoint_file(path_weight)
        assert key_weight in checkpoint, "{} does not exist as a key in {}".format(
            key_weight, path_weight)
        network.load_state_dict(checkpoint[key_weight])
        return network

    def get_classobject_from_file(self, filepath: Path, classname: str) -> object:
        if not filepath.is_absolute():
            filepath = relative2absolute(filepath)
        modulename = self.get_modulename_from_python_file(filepath)
        dir_path = filepath.parent
        self.import_module_from_file(dir_path, modulename)

        target_classobject = self.get_classobject(modulename, classname)
        return target_classobject

    def get_arguments_from_file(self, filepath: Path, classname: str, method: str) -> List[str]:
        class_object = self.get_classobject_from_file(filepath, classname)
        return get_arguments(class_object, method)

    def import_module_from_file(self, dir_path: Path, modulename: str) -> None:
        relative_dir_path_str = self.get_relative_path(dir_path)
        sys.path.append(relative_dir_path_str)
        __import__(modulename)

    @staticmethod
    def get_relative_path(path: Path) -> str:
        # no exact equivalent in Path object
        return os.path.relpath(path, start=os.getcwd())

    @staticmethod
    def get_modulename_from_python_file(filepath: Path) -> str:
        return filepath.name.replace(".py", "")

    @staticmethod
    def get_classobject(modulename: str, classname: str) -> object:
        classes = inspect.getmembers(sys.modules[modulename], inspect.isclass)
        target_class = [c[1] for c in classes if c[0] == classname]
        if len(target_class) == 0:
            raise ValueError(
                "{} does not eixst in {}.".format(classname, modulename))
        else: # inspect package retrieve one class if multiple classes exist with the same name
            return target_class[0]

    @staticmethod
    def is_torchvision_architecture(arch_name: str) -> bool:
        return hasattr(models, arch_name)
