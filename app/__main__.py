# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                                    app/__main__.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
from funcs.functions                import Functions
from noise.add_noise                import AddNoise
from graph.noise_demonstration      import NoiseDemo

from models.polynomial_regression   import PolyRegression


import warnings
import numpy as np

from typing import Any

warnings.filterwarnings("ignore")

# | VARS |-------------------------------------------------------------------------------------------------------------|
x_min: float = -3
x_max: float = 3
step : float = 0.01

noise_range: float = 3

cumulative_noise_generation_times: int = 5
# |--------------------------------------------------------------------------------------------------------------------|

x: np.ndarray = np.arange(x_min, x_max, step)

# | FUNCTION |---------------------------------------------------------------------------------------------------------|
inputs: tuple[np.ndarray, Any] = (x,) # function inputs here

def func(*args) -> np.ndarray:
    return Functions.modulated_normal(*args)
# |--------------------------------------------------------------------------------------------------------------------|

# |====================================================================================================================|
# |====================================================================================================================|

y: np.ndarray = func(*inputs)

# Noise Generation
add_noise: AddNoise = AddNoise(func)
add_noise.func_inputs(*inputs)
add_noise.noise_range(noise_range)
x_c, y_c = add_noise.cumulative_func_noise(cumulative_noise_generation_times)

noise_demo: NoiseDemo = NoiseDemo(func)
noise_demo.func_inputs(*inputs)
noise_demo.noise_range(noise_range)
noise_demo.demo(cumulative_noise_generation_times)


# Polynomial Regression
# poly: PolyRegression = PolyRegression(x, y)
# poly.noised_data(x_c, y_c)
# y_poly, r2, deg = poly.poly_optimizer()


