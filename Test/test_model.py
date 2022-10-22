from pathlib import Path

import torch
import pytest

from torchfetch.model.load import (PrivateModelLoader, get_pretrained_network, instantiate_network,
                                   get_checkpoint, get_model_state_dict,
                                   get_optimizer_state_dict, get_hyperparam, get_preprocess_name_used_for_train)

from torchfetch.model.validate import is_valid_architecture, is_valid_recipe


class TestModelLoad:
    def test_instantiate_network_public_architecture(self):
        # torchvision network
        assert isinstance(instantiate_network("resnet50"), torch.nn.Module)
        assert isinstance(instantiate_network(
            "resnet50", **{}), torch.nn.Module)
        assert isinstance(instantiate_network(
            "resnet50", **{"num_classes": 10}), torch.nn.Module)

    def test_instantiate_network_private_architecture(self):
        # public architecture defined in recipe
        assert isinstance(instantiate_network("Test/objects/recipe/public_arch.json"), torch.nn.Module)
        assert isinstance(instantiate_network("Test/objects/recipe/private_arch.json"), torch.nn.Module)

    def test_load_network_checkpoint(self):
        assert isinstance(get_pretrained_network(
            "Test/objects/recipe/private_arch.json"), torch.nn.Module)
        assert isinstance(get_checkpoint("Test/objects/recipe/private_arch.json"), dict)
        assert isinstance(get_model_state_dict("Test/objects/recipe/private_arch.json"), dict)
        assert isinstance(get_optimizer_state_dict("Test/objects/recipe/private_arch.json"), dict)
        assert get_preprocess_name_used_for_train(
            "Test/objects/recipe/private_arch.json") == "MNIST"  # retrieve tag as is

    def test_get_hyperparam(self):
        assert isinstance(get_hyperparam("Test/objects/hyperparam/example.json"), dict)

    def test_get_arguments_from_file(self):
        assert PrivateModelLoader().get_arguments_from_file(
            Path("Test/objects/architecture/feedforwardmnist.py"), "Net", "forward") == ["x"]

    @pytest.mark.xfail
    def test_get_non_existing_classobject_from_file(self):
        assert PrivateModelLoader().get_classobject_from_file(
            Path("Test/objects/architecture/feedforwardmnist.py"), "not_exists")

    @pytest.mark.xfail
    def test_get_ambiguous_classobject_from_file(self):
        assert PrivateModelLoader().get_classobject_from_file(Path(
            "Test/objects/architecture/TwoSameClassNames.py"), "Net")  # 'not_exists' class does not exist


class TestModelValidate:

    def test_is_valid_architecture(self):
        assert is_valid_architecture(
            "Test/objects/architecture/feedforwardmnist.py", classname="Net", network_kwargs={})

    def test_is_valid_pubilc_recipe(self):
        assert is_valid_recipe("Test/objects/recipe/public_arch.json")

    def test_is_valid_private_recipe(self):
        assert is_valid_recipe("Test/objects/recipe/private_arch.json")
