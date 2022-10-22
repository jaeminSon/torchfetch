from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from torchfetch.custom.utils import read_csv
from torchfetch.descriptor import DataStructureDescriptor

class CSVDataset(Dataset):

    NAME_DOWNSAMPLE_MEAN = "mean"
    NAME_DOWNSAMPLE_MEDIAN = "median"
    NAME_DOWNSAMPLE_MAX = "max"

    NAME_IMPUTE_MEAN = "mean"
    NAME_IMPUTE_MEDIAN = "median"
    NAME_IMPUTE_ZERO = "zero"
    NAME_IMPUTE_FORWARD = "forward"
    NAME_IMPUTE_BACKWARD = "backward"

    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.transform = transform

        self.path_csv = self.root / DataStructureDescriptor.NAME_CSV_FILE

        self.df = read_csv(self.path_csv)

        # read these info from root / config.yaml , for instance
        downsample_interval = 2
        input_downsample_method = "mean"
        target_downsample_method = "max"
        self.window_width = 10
        self.window_stride = 1
        self.input_column = ["\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_001_PV",
                             "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_002_PV"]
        self.target_column = [
            "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_003_PV"]
        imputations = {"\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_001_PV": "zero",
                       "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_002_PV": "mean"}
        preprocesses = {"\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_001_PV": {"mean": 1, "std": 10},
                        "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\1_AIT_002_PV": {"mean": 1, "std": 10}}

        for column, impute_method in imputations.items():
            self.impute(self.df[column], impute_method)

        self.data = self.df[self.input_column].reset_index()
        self.target = self.df[self.target_column].reset_index()

        for column, preprocess_info in preprocesses.items():
            mean = preprocess_info["mean"]
            std = preprocess_info["std"]
            self.data.loc[:, column] = (self.data[column]-mean)/std

        if downsample_interval and downsample_interval > 2:
            self.data = self.downsample(
                self.data, downsample_interval, input_downsample_method)
            self.target = self.downsample(
                self.target, downsample_interval, target_downsample_method)

        assert len(self.data) == len(self.target)
        self.len = (len(self.data) - self.window_width) // self.window_stride

    def __getitem__(self, index):
        window_start = index * self.window_stride
        window_end = window_start + self.window_width
        input_slice = self.data[window_start:window_end][self.input_column]

        # input = self.transform(window[self.input_column].to_numpy().swapaxes(0,1)) # (n_columns, window_width)
        input = input_slice.to_numpy().swapaxes(0, 1)  # (n_columns, window_width)
        if self.target_column:
            target_slice = self.target[window_start:window_end][self.target_column]
            target = target_slice.to_numpy().swapaxes(0, 1)
            return input, target
        else:
            return input

    def __len__(self) -> int:
        return self.len

    def __str__(self):
        return self.root.name

    @staticmethod
    def downsample(seq, downsample_interval: int, downsample_method: str):
        unit = "T"
        tmp_tag = "tmp_timestamp"

        seq.loc[:, tmp_tag] = CSVDataset.get_timedelta_dataframe(len(seq), unit)
        if downsample_method == CSVDataset.NAME_DOWNSAMPLE_MEAN:
            return seq.resample("{}{}".format(
                downsample_interval, unit), on=tmp_tag).mean()
        elif downsample_method == CSVDataset.NAME_DOWNSAMPLE_MAX:
            return seq.resample("{}{}".format(
                downsample_interval, unit), on=tmp_tag).max()
        elif downsample_method == CSVDataset.NAME_DOWNSAMPLE_MEDIAN:
            return seq.resample("{}{}".format(
                downsample_interval, unit), on=tmp_tag).median()

    @staticmethod
    def get_timedelta_dataframe(length, unit):
        return pd.to_timedelta(range(length), unit=unit)

    @staticmethod
    def impute(seq, imputation: str) -> pd.DataFrame:
        if imputation == CSVDataset.NAME_IMPUTE_MEAN:
            seq.fillna(seq.mean(), inplace=True)
        elif imputation == CSVDataset.NAME_IMPUTE_MEDIAN:
            seq.fillna(seq.median(), inplace=True)
        elif imputation == CSVDataset.NAME_IMPUTE_ZERO:
            seq.fillna(0, inplace=True)
        elif imputation == CSVDataset.NAME_IMPUTE_FORWARD:
            seq.fillna(method="ffill", inplace=True)
        elif imputation == CSVDataset.NAME_IMPUTE_BACKWARD:
            seq.fillna(method="bfill", inplace=True)
        else:
            raise ValueError(
                "imputation method {} does not exist. (available: mean, median, zero, forward, backward)".format(imputation))
