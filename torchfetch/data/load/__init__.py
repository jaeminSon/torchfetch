from pathlib import Path, PurePath
from typing import Callable, Dict, List, Tuple

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import MultiArgsCompose
from .dataloader import SamplerUpdatableDataLoader, DistributedSampler
from .private import get_private_dataset

from ..augment import Augmentor
from ..preprocess import Preprocessor, ToTensor

from torchfetch.custom.decorator import ignore_error
from torchfetch.custom.typing import (FilePath, Augment, DataIdentifier, SingleDataIdentifier,
                                      Preprocess)
from torchfetch.custom.utils import read_json, get_valid_kwargs
from torchfetch.descriptor import DataDescriptor, DataStructureDescriptor
from torchfetch.custom.metaclass import Singleton


__all__ = ['DataLoader', 'get_dataloader',
           'sample_single_batch', 'get_dataset_labels_by_itr']


class DataLoader(object, metaclass=Singleton):
    """
    Return pytorch dataloader specified by data, preprocess, augment
    """

    def sample_single_batch(self, data: DataIdentifier, preprocess: Preprocess, augment: Augment, task: str = None, sampler: str = None, **kwargs_dataloader) -> object:
        dataloader = self.get_dataloader(
            data, preprocess, augment, task, sampler, **kwargs_dataloader)
        for batch in dataloader:
            return batch

    def get_dataloader(self, data: DataIdentifier, preprocess: Preprocess, augment: Augment, task: str = None, sampler: str = None, **kwargs_dataloader) -> TorchDataLoader:
        data = self._rectify_data(data)
        task = self._get_task(task, data)
        augment_cls = Augmentor(augment, task)
        preprocess_cls = Preprocessor(preprocess, augment_cls.n_aug_args)
        return self._get_dataloader_from_cls(data, preprocess_cls, augment_cls, sampler, **kwargs_dataloader)

    def _rectify_data(self, data: DataIdentifier) -> DataIdentifier:
        data = self._remove_None_from_list(data)
        data = self._peel_single_element_in_list(data)
        data = self._convert2Path(data)
        return data

    def _remove_None_from_list(self, data: DataIdentifier) -> DataIdentifier:
        if type(data) == list:
            return [d for d in data if d is not None]
        else:
            return data

    def _peel_single_element_in_list(self, data: DataIdentifier) -> DataIdentifier:
        if type(data) == list and len(data) == 1:
            return data[0]
        else:
            return data

    def _convert2Path(self, data: DataIdentifier) -> DataIdentifier:
        if type(data) == list:
            return [Path(d) for d in data]
        elif type(data) == str:
            return Path(data)
        else:
            return data

    def _get_task(self, task: str, data: DataIdentifier) -> str:
        if task is None:
            return self._guess_task(data)  # None if guess fails
        else:
            return task

    @ignore_error
    def _guess_task(self, data: DataIdentifier):
        if type(data) == str:
            return DataDescriptor().get_task_from_file_structure(Path(data))
        elif isinstance(data, PurePath):
            return DataDescriptor().get_task_from_file_structure(data)
        elif type(data) == list:
            set_task = {self._guess_task(d) for d in data}
            assert len(
                set_task) == 1, "Datasets with different tasks merged. ({})".format(data)
            return set_task.pop()
        else:
            return None

    def get_data_info_from_path(self, data_path: Path) -> dict:
        path_info = data_path / DataStructureDescriptor.NAME_DATA_DESCRIPTION_FILE
        if path_info.exists():
            return read_json(path_info)
        else:
            raise IOError("No description.json found at {}.".format(
                path_info.resolve().__fspath__()))

    def get_dataset_labels_by_itr(self, data: SingleDataIdentifier, batch2labels: Callable, **kwargs_dataloader):
        assert not kwargs_dataloader["shuffle"], "Require Shuffle=False to collect labels."

        list_label = []
        dataloader = self.get_dataloader(data, None, None, **kwargs_dataloader)
        for batch in dataloader:
            list_label += batch2labels(batch)

        return list_label

    def _get_dataloader_from_cls(self, data: DataIdentifier, preprocess_cls: Preprocessor, augment_cls: Augmentor, sampler: str, **kwargs_dataloader) -> TorchDataLoader:
        if type(data) == str or isinstance(data, PurePath):
            dataset = self._get_dataset_from_path(
                Path(data), preprocess_cls, augment_cls)
        elif type(data) == list:
            dataset = self._get_multiple_dataset(
                data, preprocess_cls, augment_cls)
        else:
            raise ValueError("Data should be str or Path object.")

        return self.build_dataloader(dataset, sampler, **kwargs_dataloader)

    def _get_multiple_dataset(self, list_data: list, preprocess: transforms, augment: transforms) -> TorchDataLoader:

        class_to_idx = self.generate_class_to_idx(list_data)

        list_datasets = []
        for data in list_data:
            if type(data) == str:
                dataset = self._get_dataset_from_name(
                    data, preprocess, augment, class_to_idx)
            elif isinstance(data, PurePath):
                dataset = self._get_dataset_from_path(
                    data, preprocess, augment, class_to_idx)
            else:
                raise ValueError("Data should be str or Path object.")
            list_datasets.append(dataset)

        dataset = ConcatDataset(list_datasets)
        return dataset

    def generate_class_to_idx(self, list_data: List[SingleDataIdentifier]) -> Dict[str, int]:
        list_date_classinfo = self._get_date_classinfo_from_list_data(
            list_data)
        list_classinfo_sorted_by_date = [el[1] for el in sorted(
            list_date_classinfo, key=lambda x:x[0])]
        return self._get_dict_class_to_idx(list_classinfo_sorted_by_date)

    def _get_dict_class_to_idx(self, list_classinfo: List[List[str]]) -> Dict[str, int]:
        dict_class_to_idx = {}
        for class_info in list_classinfo:
            class_info.sort()  # same method as torchvision.datasets.DatasetFolder
            # include new class at the end if not exists
            for cls in class_info:
                if cls not in dict_class_to_idx:
                    dict_class_to_idx[cls] = len(dict_class_to_idx)
        return dict_class_to_idx

    def _get_date_classinfo_from_list_data(self, list_data: List[SingleDataIdentifier]) -> List[Tuple[str, List]]:
        # date is given by string in yyyymmdd format (e.g. 20220708)
        list_date_classinfo = []
        for data in list_data:
            data_info = self._get_classinfo_from_data(data)
            list_date_classinfo.append(
                (data_info[DataDescriptor.KEY_DATE], list(data_info[DataDescriptor.KEY_CLASS_INFO].keys())))
        return list_date_classinfo

    def _get_classinfo_from_data(self, data: FilePath) -> dict:
        if type(data) == str:
            path_data = Path(data)
        elif isinstance(data, PurePath):
            path_data = data
        else:
            raise ValueError("Data should be str or Path object.")
        return self.get_data_info_from_path(path_data)

    def _get_dataset_from_path(self, datapath: Path, preprocess: transforms, augment: transforms, class_to_idx: Dict[str, int] = None) -> Dataset:
        # only private dataset can be loaded from path (public dataset file structures not covered)
        transform = self.build_transform(preprocess, augment)
        return get_private_dataset(datapath, transform, class_to_idx)

    @staticmethod
    def build_transform(preprocess: transforms, augment: transforms) -> transforms.Compose:
        n_transform_args = augment.n_aug_args
        to_tensor = ToTensor(n_transform_args)
        list_transform = [augment] + [to_tensor] + [preprocess]
        if n_transform_args < 2:
            return transforms.Compose(list_transform)
        else:
            return MultiArgsCompose(list_transform)

    @staticmethod
    def build_dataloader(dataset: Dataset, sampler: str, **kwargs) -> TorchDataLoader:
        valid_kwargs = get_valid_kwargs(kwargs, TorchDataLoader, "__init__")
        if sampler is None:
            return TorchDataLoader(dataset, **valid_kwargs)
        elif sampler in SamplerUpdatableDataLoader.UpdatableRandomSampler:
            return SamplerUpdatableDataLoader(dataset, sampler, **valid_kwargs)
        elif sampler == DistributedSampler.NAME_DistributedSampler:
            return TorchDataLoader(dataset, sampler=DistributedSampler(dataset,
                                                                       shuffle=kwargs["shuffle"] if "shuffle" in kwargs else True,
                                                                       rank=kwargs["rank"] if "rank" in kwargs else None,
                                                                       seed=kwargs["seed"] if "seed" in kwargs else 0), **valid_kwargs)


get_dataloader = DataLoader().get_dataloader
sample_single_batch = DataLoader().sample_single_batch
get_dataset_labels_by_itr = DataLoader().get_dataset_labels_by_itr
