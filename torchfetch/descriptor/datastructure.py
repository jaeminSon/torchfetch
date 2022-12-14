import os
from pathlib import Path

from torchfetch.custom.metaclass import Singleton
from torchfetch.custom.typing import FilePath
from torchfetch.custom.utils import read_json, read_csv

__all__ = ['DataStructureDescriptor']


class DataStructureDescriptor(object, metaclass=Singleton):
    NAME_CSV_FILE = "data.csv"
    NAME_IMAGE_DIR = "image"
    NAME_ANNOT_DIR = "annotation"
    NAME_MASK_DIR = "mask"
    NAME_INFERENCE_DIR = "inference"

    ANNOT_EXT_TO_FILE_NAME = {
        "json": "annotation.json", "csv": "annotation.csv"}
    IMG_CLS_ANNOT_INPUT_TAG = "filename"
    IMG_CLS_ANNOT_TARGET_TAG = "annotation"

    NAME_DATA_DESCRIPTION_FILE = "description.json"

    ALLOWED_IMG_EXTENSION = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def img_mask(self, root: Path) -> bool:
        """ <root>/image (valid image directory), <root>/mask (valid image directory), no other valid image directories """
        return self.valid_img_folder(self.get_img_dir(root)) and self.valid_img_folder(self.get_mask_dir(root)) and self.get_num_valid_img_directories(root) == 2

    def csv(self, root: Path) -> bool:
        """ <root>/data.csv (file) """
        path_csv = root / DataStructureDescriptor.NAME_CSV_FILE
        return path_csv.exists()

    def img_cls_annotation(self, root: Path) -> bool:
        """ <root>/image (valid image directory), <root>/annotation/annotation.json, (no other valid image directories)
            or <root>/image (valid image directory), <root>/annotation/annotation.csv (no other valid image directories) """
        if self.valid_img_folder(self.get_img_dir(root)) and self.get_num_valid_img_directories(root) == 1:
            home_annot = self.get_annot_dir(root)
            for format, fname in DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME.items():
                path_annot = home_annot / fname
                if path_annot.exists() and self._valid_cls_annotation(path_annot, format):
                    return True
        return False

    def detection(self, root: Path) -> bool:
        """ <root>/image (valid image directory), <root>/annotation/annotation.json (valid detection annotation), no other valid image directories """
        dir_image = self.get_img_dir(root)
        path_annot = self.get_annot_dir(root) / \
            DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME["json"]
        return self.valid_img_folder(dir_image) and self.get_num_valid_img_directories(root) == 1 and self.valid_det_annotation(path_annot)

    def img_inference(self, root: Path) -> bool:
        """ <root>/inference (folder), 1 folder with an image """
        return self.valid_img_folder(self.get_inference_dir(root)) and self.get_num_valid_img_directories(root) == 1

    def img_cls_folder(self, root: Path) -> bool:
        """ <root>/<category1> (folder with image), <root>/<category2> (folder with image), ...
            not image-annotation, not image-mask, not detection, not image-inference, not anomaly-detection, at least 1 folder with an image """
        return not self.img_cls_annotation(root) and not self.img_mask(root) and not self.detection(root) and not self.img_inference(root) and not self.img_anomaly(root) and self.get_num_valid_img_directories(root) > 0

    def img_anomaly(self, root: Path) -> bool:
        """ <root>/image (valid image directory), not image-annotation, not image-mask, not detection, not image-inference, no other valid image directories """
        return not self.img_cls_annotation(root) and not self.img_mask(root) and not self.detection(root) and not self.img_inference(root) and self.valid_img_folder(self.get_img_dir(root)) and self.get_num_valid_img_directories(root) == 1

    def _valid_cls_annotation(self, path_annotation: FilePath, format: str) -> bool:
        """ encode rules for classification annotations """
        if format == "csv":  # check column names
            annotation = read_csv(path_annotation)
            return set(annotation.columns) == set([DataStructureDescriptor.IMG_CLS_ANNOT_INPUT_TAG, DataStructureDescriptor.IMG_CLS_ANNOT_TARGET_TAG])
        elif format == 'json':  # check if dictionary
            annotation = read_json(path_annotation)
            return type(annotation) == dict and not self.valid_det_annotation(path_annotation)

    def get_img_dir(self, root: Path) -> Path:
        return root / DataStructureDescriptor.NAME_IMAGE_DIR

    def get_mask_dir(self, root: Path) -> Path:
        return root / DataStructureDescriptor.NAME_MASK_DIR

    def get_annot_dir(self, root: Path) -> Path:
        return root / DataStructureDescriptor.NAME_ANNOT_DIR

    def get_inference_dir(self, root: Path) -> Path:
        return root / DataStructureDescriptor.NAME_INFERENCE_DIR

    def get_fpaths_detection(self, root) -> Path:
        return self.get_img_dir(root), self.get_annot_dir(root) / DataStructureDescriptor.ANNOT_EXT_TO_FILE_NAME["json"]

    def valid_det_annotation(self, filepath: Path) -> bool:
        if filepath.exists():
            label = read_json(filepath)
            return "info" in label and "categories" in label and "images" in label and "annotations" in label
        else:
            return False
    
    def get_num_valid_img_directories(self, root: Path) -> int:
        return len([p for p in root.iterdir() if self.valid_img_folder(p)])

    @staticmethod
    def valid_img_folder(root: Path) -> bool:
        
        def contain_image(root:Path) -> bool:
            for p in root.iterdir():
                if os.path.splitext(p.name)[1].lower() in DataStructureDescriptor.ALLOWED_IMG_EXTENSION:
                    return True
            return False
        
        return root.is_dir() and contain_image(root)
