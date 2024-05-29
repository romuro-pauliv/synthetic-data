# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                             app/noise/add_noise.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
import numpy as np

from typing import Callable, Any
# |--------------------------------------------------------------------------------------------------------------------|


class AddNoise(object):
    def __init__(self, func: Callable[[Any], np.ndarray]) -> None:
        """
        Initialize the AddNoise class with a function that generates an ndarray.

        Args:
            func (Callable[[Any], np.ndarray]): A function that takes any arguments and returns a numpy ndarray.
        """
        self.func: Callable[[Any], np.ndarray] = func
    
    def func_inputs(self, *args) -> None:
        """
        Store the inputs for the function.

        Args:
            *args: Variable length argument list that will be passed to the function.
        """
        self.inputs: tuple[np.ndarray, Any] = args
    
    def noise_range(self, std: float) -> None:
        """
        Set the standard deviation for the noise.

        Args:
            std (float): The standard deviation of the noise to be added.
        """
        self.n_std: float = std
    
    def _noise(self) -> np.float64:
        """
        Generate a single noise value based on the normal distribution with mean 0 and the specified standard deviation.

        Returns:
            np.float64: A random noise value.
        """
        return np.random.normal(0, self.n_std)
        
    def func_noise(self) -> np.ndarray:
        """
        Apply the function to the stored inputs and add noise to the result.

        Returns:
            np.ndarray: The function output with added noise.
        """
        self.y: np.ndarray = self.func(*self.inputs)
        noise: np.array = np.array([self._noise() for _ in range(self.y.shape[0])])
        return self.y + noise

    def cumulative_func_noise(self, times: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate cumulative noisy function outputs over a number of iterations.

        Args:
            times (int): The number of times to apply the function and add noise.

        Returns:
            tuple[np.ndarray, np.ndarray]: The cumulative inputs and the cumulative noisy outputs concatenated.
        """
        x_cumulative: list[np.ndarray] = []
        y_cumulative: list[np.ndarray] = []
        for _ in range(times):
            x_cumulative.append(self.inputs[0])
            y_cumulative.append(self.func_noise())
        
        return np.concatenate(x_cumulative), np.concatenate(y_cumulative)
