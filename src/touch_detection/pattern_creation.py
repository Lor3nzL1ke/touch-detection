import torch
import matplotlib.pyplot as plt


class Pattern:
    def __init__(self, frequency: int, config: dict):

        self.frequency = frequency
        self.min_value = config['MIN_VALUE']
        self.max_value = config['MAX_VALUE']
        self.midpoint = config['MIDPOINT']
        self.slot_width = config['SLOT_WIDTH']
        self.total_length = config['TOTAL_LENGTH']

    def assemble(self) -> torch.Tensor:

        def generate_constant_level(start_position: float, end_position: float, level: float) -> torch.Tensor:

            start_index = int(round(start_position * self.frequency, 0))
            end_index = int(round(end_position * self.frequency, 0))

            points = level * torch.ones(end_index - start_index)

            return points

        def generate_linear_increase(start_position: float, start_level: float, end_position: float, end_level: float) -> torch.Tensor:

            slope = (end_level - start_level) / (end_position - start_position)
            intercept = start_level

            points = slope * torch.arange(pow(self.frequency, -1), end_position - start_position, pow(self.frequency, -1)) + intercept

            return points

        low_level_start = 0
        low_level_end = self.midpoint - (self.slot_width / 2)

        high_level_start = self.midpoint + (self.slot_width / 2)
        high_level_end = self.total_length

        low_level_points = generate_constant_level(low_level_start, low_level_end, self.min_value)
        high_level_points = generate_constant_level(high_level_start, high_level_end, self.max_value)
        linear_increase_points = generate_linear_increase(low_level_end, self.min_value, high_level_start, self.max_value)

        pattern_points = torch.cat((low_level_points, linear_increase_points, high_level_points))

        return pattern_points

    def plot(self) -> None:

        y_values = self.assemble()
        x_values = torch.arange(0, self.total_length, pow(self.frequency, -1))

        fig, ax = plt.subplots()
        ax.plot(x_values, y_values)
        plt.grid()
        plt.show()

        return None
