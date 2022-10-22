import os
import inspect
from collections import Counter
from pathlib import Path
from datetime import date


import numpy as np
from torchvision import datasets

from torchfetch.custom.metaclass import Singleton
from torchfetch.custom.utils import read_json, read_csv

from .datastructure import DataStructureDescriptor
from .task import TaskDescriptor

from typing import Hashable, Tuple, Dict

__all__ = ['DataDescriptor']


class DataDescriptor(object, metaclass=Singleton):

    NAME_TRAIN = "train"
    NAME_VAL = "val"

    KEY_TITLE = "title"
    KEY_DATE = "date"
    KEY_IS_PUBLIC = "is_public"
    KEY_TASK = "task"
    KEY_CLASS_INFO = "class_info"
    KEY_NUM_CLASSES = "num_classes"
    KEY_CLASS_IMBALANCE_VALUE = "class_imbalance_value"
    KEY_IS_CLASS_IMBALANCE = "is_class_imbalance"
    KEY_FEW_SHOT_VALUE = "few_shot_value"
    KEY_IS_FEW_SHOT = "is_few_shot"
    KEY_DESCRIPTION = "description"
    KEY_MEAN = "mean"
    KEY_STD = "std"

    NAME_DATA_DESCRIPTION_FILE = DataStructureDescriptor.NAME_DATA_DESCRIPTION_FILE

    TORCHVISION_DATASET = {name.lower(): name for name, v in datasets.__dict__.items() if inspect.isclass(v)}  # torchvision dataset classes

    SPLIT = {"train", "val", "test", "extra"}

    @staticmethod
    def get_original_dataname(dataname: str, split=["train","val","test"]) -> str:
        """
        dataname: <original dataname>_<split>
        >>> get_original_dataname(CIFAR10_train)
        >>> CIFAR10
        >>> get_original_dataname(CIFAR10_val)
        >>> CIFAR10

        >>> get_original_dataname(image_folder) # no <split> specified after '_'
        >>> image_folder

        torchvision dataset name
        >>> get_original_dataname(cifar10_train)
        >>> CIFAR10
        >>> get_original_dataname(mnist_train)
        >>> MNIST
        """
        head, tail = DataDescriptor.split_last_occurence(dataname, "_")
        if head.lower() in DataDescriptor.TORCHVISION_DATASET:
            return DataDescriptor.TORCHVISION_DATASET[head.lower()]
        else:
            if tail.lower() in split:
                return head
            else:
                return dataname

    @staticmethod
    def get_split(dataname: str):
        """
        dataname: <original dataname>_<split>
        >>> get_split(CIFAR10_train)
        >>> train
        >>> get_split(CIFAR10_VAL)
        >>> val
        """
        split = DataDescriptor.split_last_occurence(dataname, "_")[-1]
        if split.lower() in DataDescriptor.SPLIT:
            return split.lower()
        else:
            raise ValueError("Unknown split '{}' (allowed split: 'train', 'val', 'test', 'extra').".format(split))

    @staticmethod
    def split_last_occurence(string: str, delimeter: str) -> Tuple[str, str]:
        """
        >>> split_last_occurence("a_b_c","_")
            ('a_b', 'c')
        >>> split_last_occurence("a_b_c","m")
            ('a_b_c', '')
        """
        list_subword = string.split(delimeter)
        if len(list_subword) == 1:
            return string, ""
        else:
            return delimeter.join(list_subword[:-1]), list_subword[-1]

    def get_info_private_data_from_file_structure(self, root: Path) -> dict:
        class_info, n_class, cls_imbal_val, is_cls_imbal, few_shot_val, is_few_shot = self._extract_class_related_info_from_file_structure(root)

        return {DataDescriptor.KEY_TITLE: root.name,
                DataDescriptor.KEY_TASK: self.get_task_from_file_structure(root),
                DataDescriptor.KEY_DESCRIPTION: "",
                DataDescriptor.KEY_DATE: date.today().strftime("%Y%m%d"),
                DataDescriptor.KEY_IS_PUBLIC: False,
                DataDescriptor.KEY_CLASS_INFO: class_info,
                DataDescriptor.KEY_NUM_CLASSES: n_class,
                DataDescriptor.KEY_CLASS_IMBALANCE_VALUE: cls_imbal_val,
                DataDescriptor.KEY_IS_CLASS_IMBALANCE: is_cls_imbal,
                DataDescriptor.KEY_FEW_SHOT_VALUE: few_shot_val,
                DataDescriptor.KEY_IS_FEW_SHOT: is_few_shot}

    def _extract_class_related_info_from_file_structure(self, root: Path) -> Tuple:
        dict_class_info = self.get_class_info_from_file_structure(root)

        if dict_class_info:
            num_classes = self.get_num_classes(dict_class_info)
            cls_imbal_val, is_cls_imbal = self.measure_class_imbalanceness(
                dict_class_info)
            few_shot_val, is_few_shot = self.measure_few_shotness(
                dict_class_info)
        else:
            num_classes = None
            cls_imbal_val, is_cls_imbal = None, False
            few_shot_val, is_few_shot = None, False

        return dict_class_info, num_classes, cls_imbal_val, is_cls_imbal, few_shot_val, is_few_shot

    def get_class_info_from_file_structure(self, root: Path) -> Dict[str, int]:
        if DataStructureDescriptor().csv(root):
            return self._get_class_info_csv(root)
        elif DataStructureDescriptor().img_anomaly(root):
            return self._get_class_info_img_anomaly(root)
        elif DataStructureDescriptor().img_cls_annotation(root):
            return self._get_class_info_img_cls_annotation(
                root)
        elif DataStructureDescriptor().img_mask(root):
            return self._get_class_info_img_mask(root)
        elif DataStructureDescriptor().detection(root):
            return self._get_class_info_detection(root)
        elif DataStructureDescriptor().img_cls_folder(root):
            return self._get_class_info_img_cls_folder(root)
        else:
            raise OSError("Unknown data structure (root: {})".format(root))

    def get_task_from_file_structure(self, root: Path) -> str:
        if DataStructureDescriptor().csv(root):
            return TaskDescriptor.NAME_REGRESSION
        elif DataStructureDescriptor().img_anomaly(root):
            return TaskDescriptor.NAME_ANOMALY_DETECTION
        elif DataStructureDescriptor().img_cls_annotation(root):
            return TaskDescriptor.NAME_CLASSIFICATION
        elif DataStructureDescriptor().img_mask(root):
            return TaskDescriptor.NAME_SEGMENTATION
        elif DataStructureDescriptor().detection(root):
            return TaskDescriptor.NAME_DETECTION
        elif DataStructureDescriptor().img_cls_folder(root):
            return TaskDescriptor.NAME_CLASSIFICATION
        else:
            raise OSError("Unknown data structure (root: {})".format(root))

    def _get_class_info_csv(self, root) -> Dict[str, int]:
        # TODO discussion needed (which column is target?)
        path_csv = root / DataStructureDescriptor.NAME_CSV_FILE
        df = read_csv(path_csv)
        pass

    def _get_class_info_img_anomaly(self, root: Path) -> Dict[str, int]:
        return {"normal": DataDescriptor.count_num_images(root / DataStructureDescriptor.NAME_IMAGE_DIR)}

    def _get_class_info_img_cls_annotation(self, root: Path) -> Dict[str, int]:
        home_annot = DataStructureDescriptor().get_annot_dir(root)
        for format, fname in DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME.items():
            path_annot = home_annot / fname
            if path_annot.exists():
                if format == "json":
                    d = read_json(path_annot)
                    return dict(Counter(d.values()))
                elif format == "csv":
                    df = read_csv(path_annot)
                    return dict(Counter(df[DataStructureDescriptor.IMG_CLS_ANNOT_TARGET_TAG]))

    def _get_class_info_img_cls_folder(self, root: Path) -> Dict[str, int]:
        dict_class_info = {}
        for path_category in [p for p in root.iterdir() if DataStructureDescriptor.valid_img_folder(p)]:
            dict_class_info.update(
                {path_category.name: DataDescriptor.count_num_images(path_category)})
        return dict_class_info

    def _get_class_info_img_mask(self, root: Path) -> None:
        return None  # do not count class info for segmentation tasks

    def _get_class_info_detection(self, root: Path) -> Dict[str, int]:
        path_annot = DataStructureDescriptor().get_annot_dir(root) / \
            DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME["json"]
        annot = read_json(path_annot)
        dict_id2name = {}
        for category_info in annot["categories"]:
            dict_id2name.update({category_info["id"]: category_info["name"]})

        dict_class_info = {}
        for annot in annot["annotations"]:
            name = dict_id2name[annot["category_id"]]
            dict_class_info[name] = dict_class_info.get(name, 0) + 1
        return dict_class_info

    @staticmethod
    def get_num_classes(class_info: Dict[Hashable, int]) -> int:
        return len(set(class_info.keys()))

    @staticmethod
    def count_num_images(root: Path):
        return len([p for p in root.iterdir() if os.path.splitext(p.name)[1].lower() in DataStructureDescriptor.ALLOWED_IMG_EXTENSION])

    @staticmethod
    def measure_class_imbalanceness(class_info: Dict[Hashable, int], threshold=0.08) -> Tuple[float, bool]:
        # cross-entropy
        arr_n_class = np.array(list(class_info.values()))
        n_total = sum(arr_n_class)
        if n_total > 0:
            n_class = len(arr_n_class)
            dist = arr_n_class / n_total
            dist = dist[dist != 0]
            uniform = np.array([1./n_class]*len(dist))
            val = np.sum(dist * np.log(dist / uniform))
            return val, bool(val > threshold)
        else:
            return None, False

    @staticmethod
    def measure_few_shotness(class_info: Dict[Hashable, int], threshold=20) -> Tuple[float, bool]:
        if class_info.values():
            fewshotness = min(class_info.values())
            return fewshotness, fewshotness <= threshold
        else:
            return None, False
