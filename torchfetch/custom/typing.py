from pathlib import Path
from typing import Dict, List, TypeVar

from albumentations import Compose

__all__ = ['FilePath', 'SingleDataIdentifier', 'DataIdentifier', 'Preprocess', 'Augment']

# path in str ("../<path>/<to>/<dir>") or Path
FilePath = TypeVar("FilePath", str, Path, None)

# str (public data name or path) or Path object
SingleDataIdentifier = TypeVar("SingleDataIdentifier", str, Path)
DataIdentifier = TypeVar("DataIdentifier", str, Path, List)

Preprocess = TypeVar("Preprocess", Dict, str, Path, None)

Augment = TypeVar("Augment", Compose, str, Path, None)