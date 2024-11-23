from os import path
from nptdms import TdmsFile
import torch


def fetch_data(channels: tuple, file_name: str, directory: str = "data") -> torch.Tensor:

    file_path: str = path.join("../..", directory, file_name)

    if not path.exists(file_path):
        raise Exception("Sorry, this file doesn't exist.")

    tdms_file = TdmsFile.read(file_path)
    tdms_group = tdms_file["Data"]
    tdms_data: torch.Tensor = torch.zeros(len(tdms_group["Index"]), len(channels))

    channel_index = 0

    for tdms_channel in tdms_group.channels():
        if tdms_channel.name in channels:
            tdms_data[:, channel_index] = torch.from_numpy(tdms_channel[:])
            channel_index += 1

    if channel_index < len(channels):
        raise Exception("Sorry, one or more channels could not be found.")

    return tdms_data
