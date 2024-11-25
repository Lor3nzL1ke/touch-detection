import torch


class Pattern:
    def __init__(self, frequency: int, level_low: float, level_high: float, horizontal_midpoint: float,
                 length_slot: float, length_pattern: float):

        self.frequency = frequency
        self.level_low = level_low
        self.level_high = level_high
        self.horizontal_midpoint = horizontal_midpoint
        self.length_slot = length_slot
        self.length_pattern = length_pattern

    def assemble(self) -> torch.Tensor:

        def generate_constant_level(start_position: float, end_position: float, level: float) -> torch.Tensor:

            start_index = int(round(start_position * self.frequency, 0))
            end_index = int(round(end_position * self.frequency, 0))

            constant_level = level * torch.ones(end_index - start_index - 1)

            return constant_level

        def generate_linear_increase(start_position: float, start_level: float, end_position: float, end_level: float) -> torch.Tensor:

            slope = (end_level - start_level) / (end_position - start_position)
            intercept = start_level

            linear_increase = slope * torch.range(0, end_position - start_position, pow(self.frequency, -1)) + intercept
            # torch.range() may be removed in future pytorch releases! Look into switching to torch.arange()

            return linear_increase

        level_low_start = 0
        level_low_end = self.horizontal_midpoint - (self.length_slot / 2)

        level_low_points = generate_constant_level(level_low_start, level_low_end, self.level_low)

        level_high_start = self.horizontal_midpoint + (self.length_slot / 2)
        level_high_end = self.length_pattern

        level_high_points = generate_constant_level(level_high_start, level_high_end, self.level_high)

        linear_increase_points = generate_linear_increase(level_low_end, self.level_low, level_high_start, self.level_high)

        pattern_points = torch.cat((level_low_points, linear_increase_points, level_high_points))

        return pattern_points

    def plot(self):

        return 0
