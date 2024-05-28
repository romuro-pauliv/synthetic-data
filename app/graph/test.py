from typing import Callable, Any

import numpy as np
from noise.add_noise import AddNoise

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.stats import gaussian_kde


class NoiseDemo(AddNoise):
    def __init__(self, func: Callable[[Any], np.ndarray]) -> None:
        super().__init__(func)
        
        self.std_mod: list[float] = [0.01, 0.1, 0.5, 1]
        self.bins: int = 300

        self._fig()
        
    def _fig(self) -> None:
        self.all_fig: tuple[Figure, tuple[tuple[Axes]]] = plt.subplots(
            3, len(self.std_mod), figsize=(16, 12)
        )
    
    def _save_real_std_noise(self) -> None:
        self.n_std_real: float = self.n_std
    
    def _update_std_noise(self, mod: float) -> None:
        self.n_std: float = self.n_std_real*mod
    
    def _calc_y_real(self, x_cum: np.ndarray) -> None:
        new_input: tuple[np.ndarray, Any] = tuple([x_cum]) + self.inputs[1::]
        self.y_real: np.ndarray = self.func(*new_input)
    
    def _max_noise_range(self) -> tuple[float]:
        std_range: float = (self.n_std_real*self.std_mod[-1]*4)
        return -std_range, std_range
    
    def _gaussian_kde_method(self, noise: np.ndarray, x_noise_range: np.ndarray) -> np.ndarray:
        kernel: gaussian_kde = gaussian_kde(noise)
        return kernel(x_noise_range)
    
    def _gaussian_function(self, x: np.ndarray) -> np.ndarray:
        return (1/(self.n_std*np.sqrt(2*np.pi)))*np.e**(-0.5*((x)/self.n_std)**2)
    
    def _graph_distribution_noise(self, x_noise_range: np.ndarray, pdf: np.ndarray, index: int) -> None:
        real_normal: np.ndarray = self._gaussian_function(x_noise_range)
        self.all_fig[1][1][index].plot(x_noise_range, pdf, color="purple")
        self.all_fig[1][1][index].grid(visible=True, axis="both")
        self.all_fig[1][1][index].plot(x_noise_range, real_normal, color="blue", linestyle="dotted")
        
        self.all_fig[1][2][index].plot(x_noise_range, (pdf-real_normal))
        self.all_fig[1][2][index].grid(True, "both")
    
    def _demo_noise_distribution(self, y_cum: np.ndarray, index: int) -> None:
        noise: np.ndarray = self.y_real - y_cum
        noise_min, noise_max = self._max_noise_range()
        x_noise_range: np.ndarray = np.arange(noise_min, noise_max, 0.01)
        
        pdf: np.ndarray = self._gaussian_kde_method(noise, x_noise_range)
        self._graph_distribution_noise(x_noise_range, pdf, index)
    
    def _graph_dist2d(self, x_cum: np.ndarray, y_cum: np.ndarray, index: int) -> None:
        self.all_fig[1][0][index].hist2d(x_cum, y_cum, bins=self.bins)
    
    def demo(self, times: int) -> None:
        self._save_real_std_noise()
        
        for idx, mod in enumerate(self.std_mod):
            self._update_std_noise(mod)
            x_cum, y_cum = self.cumulative_func_noise(times)
            self._calc_y_real(x_cum) if idx == 0 else None
            self._demo_noise_distribution(y_cum, idx)
            self._graph_dist2d(x_cum, y_cum, idx)

        plt.show()
        plt.clf()
            