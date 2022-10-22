from pathlib import Path

import albumentations as A
import pytest

from torchfetch.descriptor import AugmentDescriptor, DataDescriptor, RecipeDescriptor


class TestAugmentDescriptor:

    def test_infer_input_type_from_augment(self):
        assert AugmentDescriptor().infer_input_type_from_augment("Test/objects/augment/cocodetection.json") == "image"  # path
        assert AugmentDescriptor().infer_input_type_from_augment(A.load("Test/objects/augment/cocodetection.json")) == "image"  # albumentation object

        assert AugmentDescriptor().infer_input_type_from_augment("Test/objects/augment/imagenet.json") == "image"

    def test_is_detection_task(self):
        assert AugmentDescriptor().is_detection_task("Test/objects/augment/cocodetection.json")  # path
        assert AugmentDescriptor().is_detection_task(A.load("Test/objects/augment/cocodetection.json"))  # albumentation object

        assert not AugmentDescriptor().is_detection_task("Test/objects/augment/imagenet.json") # imagenet augmentation
        assert not AugmentDescriptor().is_detection_task(1) # invalid argument


class TestDataDescriptor:
    def test_split_last_occurence(self):
        assert DataDescriptor.split_last_occurence("a_a_a", "_") == ("a_a", "a")
        assert DataDescriptor.split_last_occurence("a_b_c", "_") == ("a_b", "c")
        assert DataDescriptor.split_last_occurence("a_b_c", "*") == ("a_b_c", "")

    def test_get_original_dataname(self):
        assert DataDescriptor.get_original_dataname("CIFAR10_train") == "CIFAR10"
        assert DataDescriptor.get_original_dataname("cifar10_val") == "CIFAR10"
        assert DataDescriptor.get_original_dataname("MNIST_train") == "MNIST"
        assert DataDescriptor.get_original_dataname("mnist_val") == "MNIST"
        assert DataDescriptor.get_original_dataname("ImageNet_test") == "ImageNet"
        assert DataDescriptor.get_original_dataname("imagenet_test") == "ImageNet"

    def test_get_split(self):
        assert DataDescriptor.get_split("CIFAR10_train") == "train"
        assert DataDescriptor.get_split("cifar10_VAL") == "val"
        assert DataDescriptor.get_split("cifar10_Test") == "test"
    
    @pytest.mark.xfail
    def test_get_unknown_split(self):
        DataDescriptor.get_split("CIFAR10_unknown")

    def test_measure_class_imbalanceness(self):
        assert DataDescriptor.measure_class_imbalanceness({}) == (None, False)
        assert DataDescriptor.measure_class_imbalanceness({'cat': 1}) == (0, False)
        assert DataDescriptor.measure_class_imbalanceness({'cat': 1, 'dog': 1}) == (0, False)

    def test_measure_few_shotness(self):
        assert DataDescriptor.measure_few_shotness({}) == (None, False)
        assert DataDescriptor.measure_few_shotness({'cat': 1, 'dog': 2}) == (1, True)

    @pytest.mark.parametrize("dataname", ["image_folder", "image_json", "image_csv", "image_anomaly", "segmentation", "detection1", "detection2"])
    def test_get_info_private_data_from_file_structure(self, dataname):
        info = DataDescriptor().get_info_private_data_from_file_structure(Path("Test/objects/data") / dataname)
        assert set(['title', 'task', 'description', 'date', 'is_public', 'class_info', 'num_classes',
                    'class_imbalance_value', 'is_class_imbalance', 'few_shot_value', 'is_few_shot']).issubset(set(info.keys()))
        assert info["title"] == dataname  # same root dirname
        assert not info["is_public"]

    def test_get_task_from_file_structure(self):
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/image_folder")) == "classification"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/image_json")) == "classification"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/image_csv")) == "classification"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/image_anomaly")) == "anomaly-detection"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/segmentation")) == "segmentation"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/detection1")) == "object-detection"
        assert DataDescriptor().get_task_from_file_structure(Path("Test/objects/data/detection2")) == "object-detection"

    def test_get_class_info_from_file_structure(self):
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/image_folder")) == {'cat': 1, 'dog': 1}
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/image_json")) == {'cat': 1, 'dog': 1}
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/image_csv")) == {'cat': 1, 'dog': 1}
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/image_anomaly")) == {'normal': 2}
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/detection1")) == {'motorcycle': 1, 'person': 2, 'bicycle': 1}
        assert DataDescriptor().get_class_info_from_file_structure(Path("Test/objects/data/detection2")) == {'person': 1, 'knife': 1, 'cake': 1, 'sink': 1}


class TestRecipeDescriptor:
    
    def test_get_value_from_nested_dict(self):
        assert RecipeDescriptor.get_value_from_nested_dict({0:{1:{2:{3:{4:{5:True}}}}}},[0,1,2,3,4,5])
        assert RecipeDescriptor.get_value_from_nested_dict({0:{1:{2:{3:{4:{5:True}}}}}},[0,1,2,3,4]) == {5:True}
    
    def test_get_utils(self):
        from torchfetch.custom.utils import read_json

        assert set(["author", "date", "task", "pretrain_method", "performance"]).issubset(set(RecipeDescriptor().get_info_from_recipe_file(Path("Test/objects/recipe/private_arch.json")).keys()))
        assert RecipeDescriptor().get_task_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "classification"
        assert RecipeDescriptor().get_data_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "$mnist*"
        assert RecipeDescriptor().get_architecture_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "Test/objects/architecture/feedforwardmnist.py" # architecture - filename
        assert RecipeDescriptor().get_date_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "20220411"
        assert RecipeDescriptor().get_author_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "john doe"
        assert RecipeDescriptor().get_comment_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "debugging"
        assert RecipeDescriptor().get_pretrainmethod_from_recipe_dict(read_json("Test/objects/recipe/private_arch.json")) == "None"

