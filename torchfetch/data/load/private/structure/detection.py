from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

from pycocotools.coco import COCO
from torchvision import datasets, transforms

from ...utils import MultiArgsCompose

from torchfetch.descriptor import DataStructureDescriptor

class CocoDetectionMergeable(datasets.CocoDetection):

    def __init__(self, root: Path, transform: Callable, class_to_idx: Dict[str, int]):
        image_folder, path_annotation = DataStructureDescriptor().get_fpaths_detection(root)
        root = image_folder
        annFile = path_annotation
        aug_setting = {}
        aug_setting.update({"transforms": transform}) # argument for augmentation: torchvision (https://pytorch.org/vision/stable/datasets.html)

        super(datasets.CocoDetection, self).__init__(root, **aug_setting)

        self.coco = COCOMergeable(annFile, class_to_idx)
        self.ids = list(sorted(self.coco.imgs.keys()))


class COCOMergeable(COCO):
    def __init__(self, annotation_file=None, class_to_idx: Dict[str, int] = None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        import json
        import time

        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(
                type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex(class_to_idx)

    def createIndex(self, class_to_idx: Dict[str, int]):
        # create index
        print('creating index...')
        ori_cats, anns, cats, imgs = {}, {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                ori_cats[cat['id']] = {k: v for k, v in cat.items()}

        if 'annotations' in self.dataset:
            if class_to_idx is None:
                for ann in self.dataset['annotations']:
                    imgToAnns[ann['image_id']].append(ann)
                    anns[ann['id']] = ann
            else:
                for ann in self.dataset['annotations']:
                    new_id = class_to_idx[ori_cats[ann["category_id"]]["name"]]
                    ann['category_id'] = new_id
                    imgToAnns[ann['image_id']].append(ann)
                    anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            if class_to_idx is None:
                for cat in self.dataset['categories']:
                    cats[cat['id']] = cat
            else:
                for cat in self.dataset['categories']:
                    new_id = class_to_idx[ori_cats[cat["id"]]["name"]]
                    cat['id'] = new_id
                    cats[new_id] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            if class_to_idx is None:
                for ann in self.dataset['annotations']:
                    catToImgs[ann['category_id']].append(ann['image_id'])
            else:
                for ann in self.dataset['annotations']:
                    # category_id already changed
                    catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
