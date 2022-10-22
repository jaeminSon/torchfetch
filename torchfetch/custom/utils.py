import inspect
import json

import albumentations as A
import pandas as pd
import yaml

from pathlib import Path

from typing import List, Dict
from .typing import FilePath
from .decorator import return_true_if_pass_else_false

__all__ = ['get_arguments', 'get_valid_kwargs', 'write_json', 'read_json', 'read_yaml', 'read_csv',
           'is_json_format', 'is_yaml_format', 'is_csv_format',
           'is_albumentations', 'loadable_albumentations_file',
           'relative2absolute']


def get_arguments(class_object: object, method: str) -> List[str]:
    class_signature = inspect.signature(getattr(class_object, method))
    args = list(class_signature.parameters)
    return [arg for arg in args if arg != "self"]


def get_valid_kwargs(kwargs, class_object: object, method: str) -> Dict[str, object]:
    valid_args = get_arguments(class_object, method)
    return {k: v for k, v in kwargs.items() if k in valid_args}


def write_json(path_json: FilePath, content: object) -> None:
    with open(path_json, "w") as f:
        json.dump(content, f)


def read_json(json_file: FilePath) -> dict:
    with open(json_file, "r") as f:
        dict_info = json.load(f)
    return dict_info


def read_yaml(yaml_file: FilePath) -> dict:
    with open(yaml_file) as f:
        dict_info = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return dict_info


def read_csv(csv_file: FilePath) -> pd.DataFrame:
    return pd.read_csv(csv_file)


@return_true_if_pass_else_false
def is_json_format(file: FilePath) -> bool:
    read_json(file)


@return_true_if_pass_else_false
def is_yaml_format(file: FilePath) -> bool:
    read_yaml(file)


@return_true_if_pass_else_false
def is_csv_format(file: FilePath) -> bool:
    read_csv(file)


@return_true_if_pass_else_false
def loadable_albumentations_file(fpath: Path) -> bool:
    try:
        A.load(fpath)
    except:
        A.load(fpath, data_format="yaml")


def is_albumentations(obj) -> bool:
    return isinstance(obj, A.core.composition.Compose)


def relative2absolute(path: Path) -> Path:
    return (Path.cwd() / path).resolve()


def is_allowed_character_for_filename(c: str) -> bool:
    return c.isalpha() or c.isnumeric() or c == "." or c == "-" or c == "_"


def convert2allowedfilename(string: str) -> str:
    return "".join([c for c in string if is_allowed_character_for_filename(c)]).lower()


def is_namable(string: str) -> bool:
    return all([is_allowed_character_for_filename(c) for c in string])