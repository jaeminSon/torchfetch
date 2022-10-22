import albumentations as A
import numpy as np
from PIL import Image

from torchfetch.custom.utils import (get_arguments, is_albumentations, is_json_format, is_yaml_format)

from typing import List, Tuple
from torchfetch.custom.typing import Augment


class ImageAugmentor(object):
    def __init__(self, augment: Augment):
        if is_json_format(augment):
            self.augment = A.load(augment)
        elif is_yaml_format(augment):
            self.augment = A.load(augment, data_format="yaml")
        elif is_albumentations(augment):
            self.augment = augment
        else:
            raise ValueError("Unknown image augment type '{}'.".format(augment))

    def __call__(self):
        raise NotImplementedError

    @property
    def n_aug_args(self):
        return len(get_arguments(self, "__call__"))


class SingleImageAugmentor(ImageAugmentor):
    def __call__(self, image: Image.Image) -> np.ndarray:
        # image should be numpy array for albumentations
        img_arr = np.array(image)
        aug_img_arr = self.augment(image=img_arr)["image"]
        return aug_img_arr


class ImageMaskAugmentor(ImageAugmentor):

    def __call__(self, image: Image.Image, mask: Image.Image) -> List[np.ndarray]:
        img_arr = np.array(image)
        mask_arr = np.array(mask)
        aug_result = self.augment(image=img_arr, mask=mask_arr)
        return aug_result['image'], aug_result['mask']


class ImageBboxAugmentor(ImageAugmentor):
    def __call__(self, image: Image.Image, annots: dict) -> Tuple[np.ndarray, list]:
        # image should be numpy array for albumentations
        img_arr = np.array(image)

        image_ids = [el["image_id"] for el in annots]
        ids = [el["id"] for el in annots]
        bboxes = [el["bbox"] for el in annots]
        class_labels = [el["category_id"] for el in annots]

        aug_result = self.augment(
            image=img_arr, bboxes=bboxes, class_labels=class_labels)
        annotations = self.convert2coco(list_image_id=image_ids, list_ids=ids,
                                        list_bbox=aug_result["bboxes"], list_category_id=aug_result["class_labels"])
        return aug_result["image"], annotations

    @staticmethod
    def convert2coco(list_image_id, list_ids, list_bbox, list_category_id):
        l = []
        for i in range(len(list_bbox)):
            l.append({"image_id": list_image_id[i],
                      "id": list_ids[i],
                      "bbox": list_bbox[i],
                      "category_id": list_category_id[i]})
        return l
