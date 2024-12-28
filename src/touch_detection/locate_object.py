import torch
import time

from data_loader import import_constants, fetch_data, cache_data
from data_analysis import smooth_data, cross_correlate, get_linear_regression
from pattern_creation import Pattern


def run():

    constants = import_constants('constants.yaml')

    data, data_length = fetch_data(constants.CHANNELS, constants.FILE_NAME)

    data_time = data[:, 0] * pow(constants.DATA_FREQUENCY, -1)    # in seconds (1 s)
    data_force_raw = data[:, 1:4]                       # in Newtons (1 N)
    data_position_raw = data[:, 4:7] * pow(10, -3)      # in meters  (1 m)

    pattern = Pattern(constants.DATA_FREQUENCY, constants.PATTERN_CONFIG)
    pattern_points = pattern.assemble()

    window_length = 1000        # has to be even, I think? What happens if it isn't?
    regression_length = 601
    cache_length = int(constants.DATA_FREQUENCY * pattern.total_length + window_length)

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

            # extract calculations from range()
            for cache_index in range(int(regression_length + (window_length / 2) + 1), int(cache_length - (window_length / 2))):

                time_values = time_data[(cache_index - cache_length): cache_index]
                force_values = smoothed_force_data[(cache_index - cache_length): cache_index]

                parameters = get_linear_regression(time_values, force_values)






            similarity_previous = similarity_now

    end_time = time.perf_counter()

    print(f"Total elapsed time: {end_time - start_time:.2f} seconds.")
