from pathlib import Path
from typing import Dict, List, Iterable, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torchfetch.custom.utils import read_csv, read_json
from torchfetch.descriptor import DataStructureDescriptor


class ImageAnnotationDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose, class_to_idx: Dict[str, int]) -> None:
        self.root = root
        self.transform = transform

        self.image_root = self.root / DataStructureDescriptor.NAME_IMAGE_DIR
        self.annot_root = self.root / DataStructureDescriptor.NAME_ANNOT_DIR
        self.annot_path_json = self.annot_root / \
            DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME["json"]
        self.annot_path_csv = self.annot_root / \
            DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME["csv"]

        list_path_class = self._get_list_data()
        if class_to_idx is None:
            class_to_idx = self.get_class_to_idx(
                [d[1] for d in list_path_class])
        self.samples = [(d[0], class_to_idx[d[1]]) for d in list_path_class]

    def __getitem__(self, index: int) -> Tuple[Image.Image, object]:
        path, target = self.samples[index]
        sample = self.pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return self.root.name

    def _get_list_data(self) -> list:
        if self.annot_path_json.exists():
            return self._get_list_data_from_json()
        elif self.annot_path_csv.exists():
            return self._get_list_data_from_csv()
        else:
            raise OSError("No label file found at {}".format(self.annot_root))

    def _get_list_data_from_json(self) -> list:
        list_data = []
        d = read_json(self.annot_path_json)
        for filename, annot in d.items():
            list_data.append((self.image_root / filename, annot))
        return list_data

    def _get_list_data_from_csv(self) -> list:
        list_data = []
        df = read_csv(self.annot_path_csv)
        for index in range(len(df)):
            filename = df.iloc[index][DataStructureDescriptor.IMG_CLS_ANNOT_INPUT_TAG]
            annot = df.iloc[index][DataStructureDescriptor.IMG_CLS_ANNOT_TARGET_TAG]
            list_data.append((self.image_root / filename, annot))
        return list_data

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def get_class_to_idx(classes: Iterable) -> Dict[str, int]:
        return {cls_name: i for i, cls_name in enumerate(classes)}


class ImageFolder(datasets.DatasetFolder):
    def __init__(self, root: Path, transform: transforms.Compose, class_to_idx: Dict[str, int]) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform)
        if class_to_idx is None:
            classes, class_to_idx = self._find_classes(self.root)
        else:
            classes = self._get_classes(self.root)
        samples = datasets.folder.make_dataset(
            self.root, class_to_idx, DataStructureDescriptor.ALLOWED_IMG_EXTENSION, None)

        self.loader = datasets.folder.default_loader
        self.extensions = DataStructureDescriptor.ALLOWED_IMG_EXTENSION
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _get_classes(self, root: Path) -> List[str]:
        return [d.name for d in root.iterdir() if d.is_dir()]
