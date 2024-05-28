from funcs.functions                import Functions
from noise.add_noise                import AddNoise
from models.polynomial_regression   import PolyRegression

import numpy as np

from typing import Any

# | VARS |-------------------------------------------------------------------------------------------------------------|
x_min: float = 0.01
x_max: float = 10
step : float = 0.01

noise_min_range: float = -1000
noise_max_range: float = 1000

cumulative_noise_generation_times: int = 10
# |--------------------------------------------------------------------------------------------------------------------|

x: np.ndarray = np.arange(x_min, x_max, step)

# | FUNCTION |---------------------------------------------------------------------------------------------------------|
inputs: tuple[np.ndarray, Any] = (x,0.01) # function inputs here

def func(*args) -> np.ndarray:
    return Functions.test(*args)
# |--------------------------------------------------------------------------------------------------------------------|

y: np.ndarray = func(*inputs)

# Noise Generation
add_noise: AddNoise = AddNoise(func)
add_noise.func_inputs(*inputs)
add_noise.noise_range(noise_min_range, noise_max_range)
x_c, y_c = add_noise.cumulative_func_noise(cumulative_noise_generation_times)

# Polynomial Regression
poly: PolyRegression = PolyRegression(x, y)
poly.noised_data(x_c, y_c)
y_poly, r2, deg = poly.poly_optimizer()


import matplotlib.pyplot as plt
plt.hist2d(x_c, y_c, bins=300)
plt.plot(x, y, color="white")

plt.plot(x, y_poly, color="red")
plt.title(f"r2: {r2} -> deg: {deg}")

plt.show()