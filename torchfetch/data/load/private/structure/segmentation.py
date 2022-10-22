import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchfetch.descriptor import DataStructureDescriptor

class ImageMask(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform

        image_root = self.root / DataStructureDescriptor.NAME_IMAGE_DIR
        mask_root = self.root / DataStructureDescriptor.NAME_MASK_DIR

        list_img_path = list(image_root.iterdir())
        self.list_data = self._get_list_img_mask_pair(
            mask_root, list_img_path)

    def _get_list_img_mask_pair(self, root: Path, list_fpath: List[Path] = None) -> list:
        list_pair = []
        for fpath in list_fpath:
            fname = fpath.name
            list_candidate = self.get_all_path_asterisk(
                (root / (self.drop_extension(fname)+"*")).__fspath__())
            assert len(list_candidate) == 1, "Image and mask should map uniquely. Mask file matching to {}: {}".format(
                fname, list_candidate)
            list_pair.append((fpath.__fspath__(), list_candidate[0]))

        return list_pair

    def __getitem__(self, index: int) -> Tuple[Image.Image, object]:
        img_path, mask_path = self.list_data[index]
        img = self.pil_loader(img_path, mode="RGB")
        mask = self.pil_loader(mask_path, mode="L")
        img, mask = self.transform(img, mask)
        return np.array(img), np.array(mask)

    def __len__(self) -> int:
        return len(self.list_data)

    def __str__(self) -> str:
        return self.root.name

    @staticmethod
    def pil_loader(path: str, mode: str = "RGB") -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(mode)

    @staticmethod
    def get_all_path_asterisk(pattern: str):
        return glob.glob(pattern)

    @staticmethod
    def drop_extension(filename: str):
        return ".".join(filename.split(".")[:-1])
