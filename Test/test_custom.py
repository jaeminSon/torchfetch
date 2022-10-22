from torchfetch.custom import metaclass
from torchfetch.custom import utils


class TestMetaClass:
    def test_singleton(self):

        class A(metaclass=metaclass.Singleton):
            pass

        one = A()
        theother = A()
        assert one == theother


class TestUtils:

    class A:
        def a(self, arg1, *args, **kwargs):
            return None

    def test_check_class(self):
        assert TestUtils.A().a(1) is None

    def test_get_arguments(self):
        # *args -> args, **kwargs -> kwargs
        assert utils.get_arguments(TestUtils.A, "a") == ["arg1", "args", "kwargs"]

    def test_get_valid_kwargs(self):
        # *args, **kwargs not considered
        assert utils.get_valid_kwargs({"arg1": 1, "arg2": 2, **{"kwarg": 4}}, TestUtils.A, "a") == {"arg1": 1}

    def test_convert2allowedfilename(self):
        # allow alphabet, numeric, '.' , '-', '_' and convert to lowercase
        assert utils.convert2allowedfilename("!@#$%^&*()=+[]\\|{}ABC.1-2_3") == "abc.1-2_3"

    def test_is_csv_format(self):
        assert utils.is_csv_format("Test/objects/data/image_csv/annotation/annotation.csv")

