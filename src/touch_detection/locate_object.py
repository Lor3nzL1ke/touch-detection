import torch
import time

from data_loader import fetch_data, cache_data
from data_analysis import smooth_data, cross_correlate
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

    axes = [0, 1]

    similarity_now = 0
    similarity_previous = 0
    similarity_threshold = 0.995

    smoothing_coefficient_1 = 0.99
    smoothing_coefficient_2 = 0.85
    smoothing_coefficient_3 = 0.50

    start_time = time.perf_counter()

    for axis in axes:

        for time_step in range(cache_length, data_length):

            # needs better variable names to distinguish between raw data, smooth data, and cached data
            time_data, force_data, position_data = cache_data(time_step, cache_length, data_time, data_force_raw, data_position_raw)

            interval = slice(window_length / 2, cache_length - (window_length / 2))     # also needs a better name

            """
            Pattern Recognition
            """

            smoothed_force_data = smooth_data(force_data, 'exponential', smoothing_coefficient_1)

            similarity_now = cross_correlate(pattern_points, smoothed_force_data[interval, axis])

            # this logic definitely needs some explaining
            if (similarity_previous ** 2) <= similarity_threshold or (similarity_now ** 2) > similarity_threshold:
                continue

            """
            Linear Intersection
            """

            smoothed_force_data = smooth_data(force_data, 'exponential', smoothing_coefficient_2)





            similarity_previous = similarity_now

    end_time = time.perf_counter()

    print(f"Total elapsed time: {end_time - start_time:.2f} seconds.")