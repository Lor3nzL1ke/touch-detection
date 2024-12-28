import torch


def smooth_data(noisy_data: torch.Tensor, smoothing_mode: str, strength: float = 0.5) -> torch.Tensor:

    available_smoothing_modes = ['none', 'exponential']

    if smoothing_mode not in available_smoothing_modes:
        raise Exception('Unexpected smoothing mode. Available modes are: ' + ', '.join(available_smoothing_modes) + '.')

    number_of_rows = noisy_data.size(0)
    number_of_columns = noisy_data.size(1)

    smoothed_data = torch.zeros_like(noisy_data)

    if smoothing_mode == 'none':

        smoothed_data = noisy_data

    elif smoothing_mode == 'exponential':

        smoothed_data[0, :] = noisy_data[0, :]

        for column in range(0, number_of_columns):
            for row in range(1, number_of_rows):
                smoothed_data[row, column] = (strength * smoothed_data[row - 1, column]
                                              + (1 - strength) * noisy_data[row, column])

    return smoothed_data


def cross_correlate(u_values: torch.Tensor, v_values: torch.Tensor) -> float:

    u_mean = torch.mean(u_values)
    v_mean = torch.mean(v_values)

    correlation_coefficient = (torch.sum(torch.mul(u_values - u_mean, v_values - v_mean))
                               / (torch.linalg.vector_norm(u_values - u_mean) * torch.linalg.vector_norm(v_values - v_mean)))

    return correlation_coefficient


def get_linear_regression(x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:

    extended_x_values = torch.cat((torch.ones_like(x_values), x_values), 1)

    parameters = torch.linalg.lstsq(extended_x_values, y_values).solution

    return parameters
