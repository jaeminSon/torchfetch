import glob
import os
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchfetch.descriptor import DataStructureDescriptor



class ImageInference(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform

        image_root = self.root / DataStructureDescriptor.NAME_INFERENCE_DIR

        self.samples = [p for p in image_root.iterdir() if os.path.splitext(
            p.name)[1] in DataStructureDescriptor.ALLOWED_IMG_EXTENSION]

    def __getitem__(self, index: int) -> Tuple[Image.Image, object]:
        path = self.samples[index]
        sample = self.pil_loader(path)
        sample = self.transform(sample)
        return sample, path.__fspath__()

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return self.root.name
