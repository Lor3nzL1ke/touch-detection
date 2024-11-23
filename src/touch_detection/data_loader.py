from os import path
import torch
from nptdms import TdmsFile


def fetch_data(channels: tuple, file_name: str, directory: str = "data") -> torch.Tensor:

    file_path: str = path.join("../..", directory, file_name)

    if not path.exists(file_path):
        raise Exception("Sorry, this file doesn't exist.")

    tdms_file = TdmsFile.read(file_path)
    tdms_group = tdms_file["Data"]
    tdms_index = tdms_group["Index"]
    tdms_data: torch.Tensor = torch.zeros(len(tdms_index), len(channels))
    i = 0

    for tdms_channel in tdms_group.channels():
        if tdms_channel.name in channels:
            tdms_data[:, i] = torch.from_numpy(tdms_channel[:])
            i += 1

    print(tdms_data.size())
