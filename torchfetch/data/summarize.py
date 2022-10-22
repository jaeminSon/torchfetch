from pathlib import Path, PurePath

from torchfetch.custom.typing import FilePath
from torchfetch.custom.utils import write_json
from torchfetch.descriptor import DataDescriptor

__all__ = ['DescriptionWriter']


class DescriptionWriter(object):

    def write_description(self, data: FilePath) -> None:
        if type(data) == str:
            path_data = Path(data)
        elif isinstance(data, PurePath):
            path_data = data
        else:
            raise ValueError("data should be str or Path object. {} is given.".format(data))

        write_json(path_data / DataDescriptor.NAME_DATA_DESCRIPTION_FILE,
                    DataDescriptor().get_info_private_data_from_file_structure(path_data))

write_description = DescriptionWriter().write_description



