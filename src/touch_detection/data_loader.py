from os import path
import torch
from nptdms import TdmsFile


def fetch_data(channels: tuple, file_name: str, directory: str = "data") -> torch.Tensor:

    file_path: str = path.join("../..", directory, file_name)

    if not path.exists(file_path):
        raise Exception("Sorry, this file doesn't exist.")

    tdms_file = TdmsFile.read(file_path)
    tdms_group = tdms_file['Data']

    print(tdms_group.channels())
