import torch

from data_loader import fetch_data
from pattern_maker import Pattern


def run():

    file_name = "default_D18_F40.tdms"
    directory = "data"
    channel_names = ('Index', 'FX_S1Plus2_COMP_T', 'FY_S1Plus2_COMP_T',
                     'FZ_S1Plus2_COMP_T', 'POSX_T', 'POSY_T', 'POSZ_T')

    data, data_length = fetch_data(channel_names, file_name, directory)

    data_frequency = 8000                               # in Hertz  (1 Hz)

    data_time = data[:, 0] * pow(data_frequency, -1)    # in seconds (1 s)
    data_force_raw = data[:, 1:4]                       # in Newtons (1 N)
    data_position_raw = data[:, 4:7] * pow(10, -3)      # in meters  (1 m)

    pattern = Pattern(5, 0, 1, 2, 1, 4)
    points = pattern.assemble()
