from pathlib import Path

from .load import PrivateModelLoader

from torchfetch.custom.decorator import return_true_if_pass_else_false

from torchfetch.custom.typing import FilePath

from torchfetch.descriptor import RecipeDescriptor


__all__ = ['Validator']


class Validator(object):

    @return_true_if_pass_else_false
    def is_valid_architecture(self, filepath: str, classname: str, network_kwargs: dict) -> bool:
        network_class = PrivateModelLoader().get_classobject_from_file(
            Path(filepath), classname)
        network_class(**network_kwargs)

    @return_true_if_pass_else_false
    def is_valid_recipe(self, filepath: str) -> bool:
        network = PrivateModelLoader().instantiate_network_from_recipe_path(filepath)
        if RecipeDescriptor().get_checkpoint_filepath_from_recipe_file(filepath) is not None:
            network = PrivateModelLoader().load(filepath)


is_valid_architecture = Validator().is_valid_architecture
is_valid_recipe = Validator().is_valid_recipe