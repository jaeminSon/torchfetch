from pathlib import Path

from torchvision import datasets, transforms

from .classification import ImageFolder

from torchfetch.descriptor import DataStructureDescriptor


class AnomalyImageFolder(ImageFolder):
    NAME_NORMAL_CLASS = "normal"

    def __init__(self, root: Path, transform: transforms.Compose, class_normal: int) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform)
        samples = [(p, class_normal)
                   for p in (DataStructureDescriptor().get_img_dir(root)).iterdir()]

        self.loader = datasets.folder.default_loader
        self.classes = [AnomalyImageFolder.NAME_NORMAL_CLASS]
        self.samples = samples
        self.targets = [class_normal] * len(samples)