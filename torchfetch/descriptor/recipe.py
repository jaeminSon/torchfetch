from pathlib import Path
from typing import List, Optional

from torchfetch.custom.utils import read_json
from torchfetch.custom.metaclass import Singleton


class RecipeDescriptor(object, metaclass=Singleton):

    KEY_DESCRIPTION = "description"
    KEY_COMMENT = [KEY_DESCRIPTION, "comment"]
    KEY_DATA = [KEY_DESCRIPTION, "data"]
    KEY_ARCHITECTURE = "architecture"
    KEY_ARCHITECTURE_FILEPATH = [KEY_ARCHITECTURE, "filepath"]
    KEY_ARCHITECTURE_FILEPATH_CLASSNAME = [KEY_ARCHITECTURE, "classname"]
    KEY_ARCHITECTURE_ARGS = [KEY_ARCHITECTURE, "arguments"]
    KEY_CHECKPOINT = "checkpoint"
    KEY_CHECKPOINT_FILEPATH = [KEY_CHECKPOINT, "filepath"]
    KEY_WEIGHT = [KEY_CHECKPOINT, "key_weight_state_dict"]
    KEY_OPTIMIZER = [KEY_CHECKPOINT, "key_optimizer_state_dict"]
    KEY_INFORMATION = "information"
    KEY_AUTHOR = [KEY_INFORMATION, "author"]
    KEY_PERFORMANCE = [KEY_INFORMATION, "performance"]
    KEY_TASK = [KEY_INFORMATION, "task"]
    KEY_DATE = [KEY_INFORMATION, "date"]
    KEY_PRETRAIN_METHOD = [KEY_INFORMATION, "pretrain_method"]
    KEY_PREPROCESS = "preprocess"

    NAME_RECIPE_DIR = "recipe"
    NAME_DATA_DIR = "data"

    DELIMITER = "-"
    NAME_PROJECT = "project"
    NAME_DATA = "data"
    NAME_ARCHITECTURE = "architecture"
    NAME_COMMENT = "comment"
    NAME_FORMAT = DELIMITER.join(
        [NAME_PROJECT, NAME_DATA, NAME_ARCHITECTURE, NAME_COMMENT])

    def _get_value_from_recipe_file(self, path_recipe: Path, list_key: List[str]) -> object:
        dictionary = read_json(path_recipe)
        return self.get_value_from_nested_dict(dictionary, list_key)

    def get_checkpoint_filepath_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_CHECKPOINT_FILEPATH)

    def get_checkpoint_keyoptimizer_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_OPTIMIZER)

    def get_checkpoint_keyweight_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_WEIGHT)

    def get_preprocess_filepath_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, [self.KEY_PREPROCESS])

    def get_architecture_args_from_recipe_file(self, path_recipe: Path) -> dict:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_ARCHITECTURE_ARGS)

    def get_architecture_filepath_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_ARCHITECTURE_FILEPATH)

    def get_architecture_classname_from_recipe_file(self, path_recipe: Path) -> str:
        return self._get_value_from_recipe_file(path_recipe, self.KEY_ARCHITECTURE_FILEPATH_CLASSNAME)

    def get_info_from_recipe_file(self, path_recipe: Path) -> dict:
        return self._get_value_from_recipe_file(path_recipe, [self.KEY_INFORMATION])

    def get_task_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_TASK)

    def get_data_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_DATA)

    def get_architecture_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_ARCHITECTURE_FILEPATH)

    def get_date_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_DATE)

    def get_author_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_AUTHOR)

    def get_comment_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_COMMENT)

    def get_pretrainmethod_from_recipe_dict(self, dict_recipe: dict) -> str:
        return self.get_value_from_nested_dict(dict_recipe, self.KEY_PRETRAIN_METHOD)

    @staticmethod
    def get_value_from_nested_dict(dict_info: dict, list_key: list) -> Optional[dict]:
        try:
            for key in list_key:
                dict_info = dict_info[key]
            return dict_info
        except:
            return None