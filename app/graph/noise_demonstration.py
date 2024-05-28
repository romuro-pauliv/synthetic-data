# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                   app/graph/noise_demonstration.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
from noise.add_noise import AddNoise

from typing import Callable, Any

import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.figure      import Figure
from matplotlib.axes        import Axes
from scipy.stats            import gaussian_kde

from typing import Any
# |--------------------------------------------------------------------------------------------------------------------|


class NoiseDemonstration(AddNoise):
    def __init__(self, func: Callable[[Any], np.ndarray]) -> None:
        super().__init__(func)
        self.range_modificators : list[float]   = [0.1, 1, 2, 2.5]
        self.bins               : int           = 300
        
        self._fig()
        
    def _fig(self) -> None:
        self.all_fig: tuple[Figure, tuple[tuple[Axes]]] = plt.subplots(
            2, len(self.range_modificators), figsize=(16, 8)
        )
    
    def _calc_y_real(self, x_cum: np.ndarray) -> np.ndarray:
        new_input: tuple[np.ndarray, Any] = tuple([x_cum]) + self.inputs[1::]
        self.y_real: np.ndarray = self.func(*new_input)
    
    def _hist2d_graph(self, x_cum: np.ndarray, y_cum: np.ndarray, index: int) -> None:
        self.all_fig[1][0][index].hist2d(x_cum, y_cum, bins=self.bins)
    
    def _dist_noise_graph(self, x: np.ndarray, pdf: np.ndarray, index: int) -> None:
        self.all_fig[1][1][index].plot(x, pdf)
        self.all_fig[1][1][index].grid(visible=True, axis="both")
        self.all_fig[1][1][index].plot(x, (1/(2*np.pi*np.sqrt(self.n_std))*np.e**(-((x)**2)/(2*self.n_std))))
    
    def _get_range_noise_limit(self) -> tuple[float]:
        std_range: float = (self.n_std_real*self.range_modificators[-1]*4)
        return -std_range, std_range
    
    def _save_real_noise_range(self) -> None:
        self.n_std_real: float = self.n_std
    
    def _update_noise_range(self, mod: float) -> None:
        self.n_std: float = self.n_std_real*mod
    
    def _get_noise(self, y_cum: np.ndarray) -> np.ndarray:
        return self.y_real - y_cum
    
    def _demonstrate_noise_dist(self, y_cum: np.ndarray, index: int) -> None:        
        noise: np.ndarray = self._get_noise(y_cum)
        kernel: gaussian_kde = gaussian_kde(noise)
        
        min_xlim, max_xlim = self._get_range_noise_limit()
        
        x: np.ndarray = np.arange(min_xlim, max_xlim, 0.01)
        pdf: np.ndarray = kernel(x)
        
        self._dist_noise_graph(x, pdf, index)
        
    
    def demonstrate_noise(self, times: int) -> None:
        self._save_real_noise_range()
        
        for index, mod in enumerate(self.range_modificators):
            self._update_noise_range(mod)
            
            x_cum, y_cum = self.cumulative_func_noise(times)
            
            if index == 0:
                self._calc_y_real(x_cum)
            
            self._demonstrate_noise_dist(y_cum, index)
            self._hist2d_graph(x_cum, y_cum, index)
    
    def run(self) -> None:
        plt.show()
        plt.clf()