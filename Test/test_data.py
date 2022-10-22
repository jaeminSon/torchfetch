import os
from pathlib import Path

import pytest
import albumentations as A

# get_dataloader (tested by sample_single_batch)
from torchfetch.data.load import get_dataset_labels_by_itr, sample_single_batch
from torchfetch.data.validate import is_valid_augment, is_valid_preprocess


class TestDataLoad:

    @pytest.mark.parametrize("preprocess", [{"mean": [0], "std": [1]}, "Test/objects/preprocess/mnist.json", None])
    def test_load_public_data_varying_preprocess(self, preprocess):
        datum = sample_single_batch(data="Test/objects/data/image_folder", preprocess=preprocess, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                                 "pin_memory": True,
                                                                                                                                 "batch_size": 1,
                                                                                                                                 "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4 # (n, c, h, w)
        assert tuple(datum[1].shape) == (1,)

    @pytest.mark.parametrize("augment", [A.Compose([A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=1)]), "Test/objects/augment/mnist.json", None])
    def test_load_public_data_varying_augment(self, augment):
        datum = sample_single_batch(data="Test/objects/data/image_folder", preprocess=None, augment=augment, sampler=None, **{"num_workers": 1,
                                                                                                                              "pin_memory": True,
                                                                                                                              "batch_size": 1,
                                                                                                                              "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4 # (n, c, h, w)
        assert tuple(datum[1].shape) == (1,)

    def test_load_with_task_specified(self):
        datum = sample_single_batch(data="Test/objects/data/image_folder", preprocess=None, augment=None, task="classification", sampler=None, **{"num_workers": 1,
                                                                                                                                                  "pin_memory": True,
                                                                                                                                                  "batch_size": 1,
                                                                                                                                                  "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4 # (n, c, h, w)
        assert tuple(datum[1].shape) == (1,)

    @pytest.mark.xfail
    def test_load_public_data_unknown_data(self):
        # not_exists does not exist, thus fail
        sample_single_batch(data="Test/objects/not_exists", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                            "pin_memory": True,
                                                                                                            "batch_size": 1,
                                                                                                            "shuffle": False})

    def test_load_private_segmentation_data_with_augmentation(self):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/segmentation",
                                    preprocess=None,
                                    augment="Test/objects/augment/segmentation_strong.json",
                                    **{"num_workers": 1,
                                       "pin_memory": True,
                                       "batch_size": 1,
                                       "shuffle": False})
        assert len(datum) == 2  # (image_batch, mask_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert len(datum[1].shape) == 3  # (n, h, w) for mask_batch
        assert tuple(datum[0].shape)[:2] == (
            1, 3)  # (batchsize, image channels)
        assert tuple(datum[1].shape)[:1] == (1, )  # (batchsize, )

    def test_load_private_segmentation_data(self):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/segmentation", preprocess=None, augment=None,  **{"num_workers": 1,
                                                                                                         "pin_memory": True,
                                                                                                         "batch_size": 1,
                                                                                                         "shuffle": False})
        assert len(datum) == 2  # (image_batch, mask_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert len(datum[1].shape) == 3  # (n, h, w) for mask_batch
        assert tuple(datum[0].shape)[:2] == (
            1, 3)  # (batchsize, image channels)
        assert tuple(datum[1].shape)[:1] == (1, )  # (batchsize, )

    def test_load_private_data_with_path(self):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/image_csv", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                   "pin_memory": True,
                                                                                                                   "batch_size": 1,
                                                                                                                   "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert len(datum[1].shape) == 1  # (n, ) for label
        assert tuple(datum[0].shape)[:2] == (1, 3)  # (batchsize, channels)
        assert tuple(datum[1].shape) == (1, )  # (batchsize, )

    @pytest.mark.parametrize("dataname", ["Test/objects/data/image_csv", "Test/objects/data/image_json", "Test/objects/data/image_folder"])
    def test_load_private_classification_data(self, dataname):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch(dataname, preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                              "pin_memory": True,
                                                                                              "batch_size": 1,
                                                                                              "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert len(datum[1].shape) == 1  # (n, ) for label
        assert tuple(datum[0].shape)[:2] == (1, 3)  # (batchsize, channels)
        assert tuple(datum[1].shape) == (1, )  # (batchsize, )

    @pytest.mark.xfail
    def test_load_different_image_sizes_without_resize(self):
        # batch_size = 2 for images with difference sizes
        sample_single_batch("Test/objects/data/image_csv", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                           "pin_memory": True,
                                                                                                           "batch_size": 2,
                                                                                                           "shuffle": False})

    @pytest.mark.xfail
    def test_load_empty_image_folder(self):
        # file exists: empty_image_folder/cls/dummy
        sample_single_batch("Test/objects/data/empty_image_folder", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                    "pin_memory": True,
                                                                                                                    "batch_size": 2,
                                                                                                                    "shuffle": False})

    def test_load_different_image_sizes_with_resize(self):
        # batch_size = 2 resizing to the same size
        datum = sample_single_batch("Test/objects/data/image_csv",
                                    preprocess=None,
                                    augment="Test/objects/augment/imagenet.json",
                                    sampler=None, **{"num_workers": 1,
                                                     "pin_memory": True,
                                                     "batch_size": 2,
                                                     "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert len(datum[1].shape) == 1  # (n, ) for label
        assert tuple(datum[0].shape)[:2] == (2, 3)  # (batchsize, channels)
        assert tuple(datum[1].shape) == (2, )  # (batchsize, )

    @pytest.mark.parametrize("augment,", [None, "Test/objects/augment/cocodetection.json"])
    def test_load_private_detection_with_augment(self, augment):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/detection1", preprocess=None, augment=augment, sampler=None, **{"num_workers": 1,
                                                                                                                       "pin_memory": True,
                                                                                                                       "batch_size": 1,
                                                                                                                       "shuffle": False,
                                                                                                                       "collate_fn": lambda x: x})
        assert len(datum) == 1  # batchsize
        assert type(datum) == list  # [[img1, label1], [img2, label2], ...]
        assert len(datum[0][0].shape) == 3  # (c, h, w)
        assert datum[0][0].shape[0] == 3  # image channel
        assert type(datum[0][1]) == list  # label = [annot1, annot2, ...]
        assert set(['image_id', 'bbox', 'category_id', 'id']).issubset(
            set(datum[0][1][0].keys()))  # detection label

    @pytest.mark.parametrize("preprocess,", [None, "Test/objects/preprocess/imagenet.json"])
    def test_load_private_detection_with_preprocess(self, preprocess):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/detection1", preprocess=preprocess, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                          "pin_memory": True,
                                                                                                                          "batch_size": 1,
                                                                                                                          "shuffle": False,
                                                                                                                          "collate_fn": lambda x: x})
        assert len(datum) == 1  # batchsize
        assert type(datum) == list  # [[img1, label1], [img2, label2], ...]
        assert len(datum[0][0].shape) == 3  # (c, h, w)
        assert datum[0][0].shape[0] == 3  # image channel
        assert type(datum[0][1]) == list  # label = [annot1, annot2, ...]
        assert set(['image_id', 'bbox', 'category_id', 'id']).issubset(
            set(datum[0][1][0].keys()))  # detection label

    def test_load_private_inference_data(self):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/image_inference", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                         "pin_memory": True,
                                                                                                                         "batch_size": 1,
                                                                                                                         "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert tuple(datum[0].shape)[:2] == (1, 3)  # (batchsize, channels)
        assert type(datum[1]) == list or type(
            datum[1]) == tuple  # list (or tuple) of path
        assert type(datum[1][0]) == str  # path == str
        assert len(datum[1]) == 1  # length of path == batchsize

    def test_load_private_image_anomaly_data(self):
        """ combination of process and augment tested with 'test_load_public_data' function, thus set None to both """
        datum = sample_single_batch("Test/objects/data/image_anomaly", preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                       "pin_memory": True,
                                                                                                                       "batch_size": 1,
                                                                                                                       "shuffle": False})
        assert len(datum) == 2  # (image_batch, (dummy) label_batch)
        assert len(datum[0].shape) == 4  # (n, c, h, w) for image_batch
        assert tuple(datum[0].shape)[:2] == (1, 3)  # (batchsize, channels)

    @pytest.mark.parametrize("list_data,preprare_data_description",
                             [(["Test/objects/data/image_folder", "Test/objects/data/image_csv"],
                               ["Test/objects/data/image_folder", "Test/objects/data/image_csv"])],
                             indirect=["preprare_data_description"])
    def test_load_merged_data_different_file_structure(self, list_data, preprare_data_description):
        datum = sample_single_batch(data=list_data, preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                    "pin_memory": True,
                                                                                                    "batch_size": 1,
                                                                                                    "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert tuple(datum[1].shape) == (1,)

    def test_load_merged_data_with_None(self):
        datum = sample_single_batch(data=["Test/objects/data/image_folder", None], preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                                                                   "pin_memory": True,
                                                                                                                                   "batch_size": 1,
                                                                                                                                   "shuffle": False})
        assert len(datum) == 2  # (image_batch, label_batch)
        assert len(datum[0].shape) == 4 # (n, c, h, w)
        assert tuple(datum[1].shape) == (1,)

    @pytest.mark.xfail
    def test_load_merged_data_without_description_files(self):
        sample_single_batch(["Test/objects/data/detection1", "Test/objects/data/detection2"], None, None,
                            **{"num_workers": 1,
                               "pin_memory": True,
                               "batch_size": 1,
                               "shuffle": False,
                               "collate_fn": lambda x: x})

    @pytest.fixture
    def preprare_data_description(self, request):
        # before test
        from torchfetch.data import write_description
        for datapath in request.param:  # save description for all given datanames
            write_description(datapath)

        yield  # test

        # after test
        for dataname in request.param:  # remove description for all given datanames
            os.remove(Path(dataname) / "description.json")

    @pytest.mark.parametrize("list_data,preprare_data_description", [(["Test/objects/data/detection1", "Test/objects/data/detection2"], ["Test/objects/data/detection1", "Test/objects/data/detection2"])],
                             indirect=["preprare_data_description"])
    def test_load_merged_data_with_description_files(self, list_data, preprare_data_description):
        detection_dataloader_kwargs = {"num_workers": 1, "pin_memory": True,
                                       "batch_size": 1, "shuffle": False, "collate_fn": lambda x: x}
        datum = sample_single_batch(
            list_data, None, None, **detection_dataloader_kwargs)
        assert len(datum) == 1  # batchsize
        assert type(datum) == list  # [[img1, label1], [img2, label2], ...]
        assert len(datum[0][0].shape) == 3  # (c, h, w)
        assert datum[0][0].shape[0] == 3  # image channel
        assert type(datum[0][1]) == list  # label = [annot1, annot2, ...]
        assert set(['image_id', 'bbox', 'category_id', 'id']).issubset(
            set(datum[0][1][0].keys()))  # detection label

        # check label
        # detection1 annotation count: {"motorcycle": 1, "person": 2, "bicycle": 1}
        # detection2 annotation count: {"person": 1, "knife": 1, "cake": 1, "sink": 1}
        # category_id in merged datasaet:  bicycle:0, motorcycle:1, person:2, cake:3, knife:4, sink:5
        labels = get_dataset_labels_by_itr(
            list_data, lambda x: list(x[0][1]), **detection_dataloader_kwargs)
        assert set([(l["category_id"], l["id"]) for l in labels]) == set([(0, 1766676), (1, 151091),
                                                                          (2, 1260346), (2,
                                                                                         202758), (2, 455475),
                                                                          (3, 1085508), (4, 692513), (5, 1982455)])

    @pytest.mark.xfail
    def test_load_None(self):
        sample_single_batch(data=None, preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                       "pin_memory": True,
                                                                                       "batch_size": 1,
                                                                                       "shuffle": False})

    @pytest.mark.xfail
    def test_load_list_None(self):
        sample_single_batch(data=[None, None], preprocess=None, augment=None, sampler=None, **{"num_workers": 1,
                                                                                               "pin_memory": True,
                                                                                               "batch_size": 1,
                                                                                               "shuffle": False})

    def test_get_dataset_labels_by_itr(self):
        labels = get_dataset_labels_by_itr("Test/objects/data/image_folder", lambda x: x[1].tolist(), **{"num_workers": 1,
                                                                                    "pin_memory": True,
                                                                                    "batch_size": 1,
                                                                                    "shuffle": False})
        assert len(labels) == 2
        assert set(labels) == set(range(2))

    @pytest.mark.xfail
    def test_get_dataset_labels_by_itr_with_shuffle(self):
        # assert shuffle == False to get dataset labels
        get_dataset_labels_by_itr("Test/objects/data/image_folder", lambda x: list(x[0]), **{"num_workers": 1,
                                                                                             "pin_memory": True,
                                                                                             "batch_size": 1,
                                                                                             "shuffle": True})


class TestDataUpload:
    def test_validate_augment(self):
        assert is_valid_augment("Test/objects/augment/imagenet.json")
        assert is_valid_augment("Test/objects/augment/mnist.yaml")

    def test_validate_preprocess(self):
        assert is_valid_preprocess("Test/objects/preprocess/imagenet.json")
