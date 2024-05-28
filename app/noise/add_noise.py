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
        self.func: Callable[[Any], np.ndarray] = func
    
    def func_inputs(self, *args) -> None:
        self.inputs: tuple[np.ndarray, Any] = args
    
    def noise_range(self, std: float) -> None:
        self.n_std: float = std
    
    def _noise(self) -> np.float64:
        # return np.random.uniform(np.random.uniform(self.n_min, 0), np.random.uniform(0, self.n_max))
        return np.random.normal(0, self.n_std)
        
    def func_noise(self) -> np.ndarray:
        self.y: np.ndarray = self.func(*self.inputs)
        noise: np.array = np.array([self._noise() for _ in range(self.y.shape[0])])
        return self.y + noise

    def cumulative_func_noise(self, times: int) -> None:
        x_cumulative: list[np.ndarray] = []
        y_cumulative: list[np.ndarray] = []
        for _ in range(times):
            x_cumulative.append(self.inputs[0])
            y_cumulative.append(self.func_noise())
        
        return np.concatenate(x_cumulative), np.concatenate(y_cumulative)