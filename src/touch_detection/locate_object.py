import torch
import time
from data_loader import fetch_data, cache_data
from pattern_creation import Pattern


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

    pattern = Pattern(data_frequency, 0, 1.5, 0.125, 0.08, 0.25)
    pattern_points = pattern.assemble()

    window_length = 1000
    cache_length = int(data_frequency * pattern.length_pattern + window_length)

    axes = [1, 2]

    start_time = time.perf_counter()

    for axis in axes:

        for time_step in range(cache_length, data_length):

            cached_time, cached_forces_raw, cached_positions_raw = cache_data(time_step, cache_length, data_time,
                                                                              data_force_raw, data_position_raw)


    end_time = time.perf_counter()

    print(f"Total elapsed time: {end_time - start_time:.2f} seconds.")