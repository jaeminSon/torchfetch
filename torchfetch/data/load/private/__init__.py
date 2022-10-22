from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset
from torchvision import transforms

from .structure import AnomalyImageFolder, CSVDataset, ImageAnnotationDataset, ImageMask, CocoDetectionMergeable, ImageInference, ImageFolder

from torchfetch.descriptor import DataStructureDescriptor


__all__ = ['PrivateDataset', 'get_private_dataset']


class PrivateDataset(object):
    def __new__(self, root: Path, transform: transforms.Compose, class_to_idx: Dict[str, int]) -> Dataset:
        if DataStructureDescriptor().csv(root):
            return CSVDataset(root, transform)
        elif DataStructureDescriptor().img_anomaly(root):
            class_normal = class_to_idx[AnomalyImageFolder.NAME_NORMAL_CLASS] if class_to_idx else 0
            return AnomalyImageFolder(root, transform=transform, class_normal=class_normal)
        elif DataStructureDescriptor().img_cls_annotation(root):
            return ImageAnnotationDataset(root, transform=transform, class_to_idx=class_to_idx)
        elif DataStructureDescriptor().img_mask(root):
            return ImageMask(root, transform=transform)
        elif DataStructureDescriptor().detection(root):
            return CocoDetectionMergeable(root, transform=transform, class_to_idx=class_to_idx)
        elif DataStructureDescriptor().img_inference(root):
            return ImageInference(root, transform=transform)
        elif DataStructureDescriptor().img_cls_folder(root):
            return ImageFolder(root, transform=transform, class_to_idx=class_to_idx)
        else:
            raise OSError("Unknown data structure (root: {})".format(root))

def get_private_dataset(home_dir: Path, transform: transforms.Compose, class_to_idx: Dict[str, int]) -> Dataset:
    assert home_dir is not None, "Home directory should not be None"
    return PrivateDataset(root=home_dir, transform=transform, class_to_idx=class_to_idx)
