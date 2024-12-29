import torch
import time as t

from data_loader import import_constants, fetch_data, cache_data
from data_analysis import smooth_data, cross_correlate, get_linear_regression
from pattern_creation import Pattern


def run():

    # these should be moved to constants.yaml too
    window_length = 1000  # has to be even, I think? What happens if it isn't?
    regression_length = 601

    constants = import_constants('constants.yaml')

    data, data_length = fetch_data(constants.CHANNELS, constants.FILE_NAME)

    time = data[:, 0] * pow(constants.DATA_FREQUENCY, -1)   # time in seconds (1 s)
    force = data[:, 1:4]                                    # force data in Newtons (1 N)
    position = data[:, 4:7] * pow(10, -3)                   # position data in meters  (1 m)

    pattern = Pattern(constants.DATA_FREQUENCY, constants.PATTERN_CONFIG)
    pattern_points = pattern.assemble()

    cache_length = int(constants.DATA_FREQUENCY * pattern.total_length + window_length)
    interval = slice(window_length / 2, cache_length - (window_length / 2))  # needs a better name

    similarity_now = 0
    similarity_previous = 0

    start_time = t.perf_counter()

    for axis in constants.AXES:

        for time_step in range(cache_length, data_length):

            time_cached, force_cached, position_cached = cache_data(time_step, cache_length, time, force, position)

            """
            Pattern Recognition
            """

            force_cached_smoothed = smooth_data(force_cached, 'exponential', constants.SMOOTHING_FACTOR_1)

            similarity_now = cross_correlate(pattern_points, force_cached_smoothed[interval, axis])

            if not is_valid_interval(similarity_previous, similarity_now, constants.SIMILARITY_THRESHOLD):
                continue

            """
            Linear Intersection
            """

            force_cached_smoothed = smooth_data(force_cached, 'exponential', constants.SMOOTHING_FACTOR_2)

            # extract calculations from range()
            for cache_index in range(int(regression_length + (window_length / 2) - 1), int(cache_length - (window_length / 2))):

                time_values = time_cached[(cache_index - cache_length): cache_index]
                force_values = force_cached_smoothed[(cache_index - cache_length): cache_index]

                parameters = get_linear_regression(time_values, force_values)






            similarity_previous = similarity_now

    end_time = t.perf_counter()

    print(f"Total elapsed time: {end_time - start_time:.2f} seconds.")


def is_valid_interval(similarity_before: float, similarity_now: float, threshold: float) -> bool:

    # this logic definitely needs some explaining

    return (similarity_before ** 2) > threshold and not (similarity_now ** 2) > threshold
