# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                   app/graph/noise_demonstration.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
from noise.add_noise        import AddNoise

from typing                 import Callable, Any

import numpy                as np
import matplotlib.pyplot    as plt

from matplotlib.figure      import Figure
from matplotlib.axes        import Axes

from scipy.stats            import gaussian_kde
# |--------------------------------------------------------------------------------------------------------------------|


class NoiseDemo(AddNoise):
    def __init__(self, func: Callable[[Any], np.ndarray]) -> None:
        """
        Initialize the NoiseDemo class, which inherits from AddNoise.

        Args:
            func (Callable[[Any], np.ndarray]): A function that takes any arguments and returns a numpy ndarray.
        """
        super().__init__(func)
        
        self.std_mod: list[float] = [0.01, 0.1, 0.5, 1]
        self.bins: int = 300

        self._fig()
        
    def _fig(self) -> None:
        """
        Initialize the figure for plotting the noise distribution and 2D histogram.
        """
        self.all_fig: tuple[Figure, tuple[tuple[Axes]]] = plt.subplots(
            2, len(self.std_mod), figsize=(16, 8)
        )
    
    def _save_real_std_noise(self) -> None:
        """
        Save the current standard deviation of the noise.
        """
        self.n_std_real: float = self.n_std
    
    def _update_std_noise(self, mod: float) -> None:
        """
        Update the standard deviation of the noise based on a modifier.

        Args:
            mod (float): The modifier to adjust the standard deviation.
        """
        self.n_std: float = self.n_std_real * mod
    
    def _calc_y_real(self, x_cum: np.ndarray) -> None:
        """
        Calculate the real function outputs for the cumulative inputs without added noise.

        Args:
            x_cum (np.ndarray): The cumulative inputs.
        """
        new_input: tuple[np.ndarray, Any] = tuple([x_cum]) + self.inputs[1:]
        self.y_real: np.ndarray = self.func(*new_input)
    
    def _max_noise_range(self) -> tuple[float, float]:
        """
        Calculate the range for noise based on the maximum modified standard deviation.

        Returns:
            tuple[float, float]: The minimum and maximum noise range.
        """
        std_range: float = self.n_std_real * self.std_mod[-1] * 4
        return -std_range, std_range
    
    def _gaussian_kde_method(self, noise: np.ndarray, x_noise_range: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function of the noise using Gaussian KDE.

        Args:
            noise (np.ndarray): The noise data.
            x_noise_range (np.ndarray): The range of noise values.

        Returns:
            np.ndarray: The estimated probability density function.
        """
        kernel: gaussian_kde = gaussian_kde(noise)
        return kernel(x_noise_range)
    
    def _gaussian_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the Gaussian function for the noise distribution.

        Args:
            x (np.ndarray): The range of noise values.

        Returns:
            np.ndarray: The Gaussian function values.
        """
        return (1 / (self.n_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / self.n_std) ** 2)
    
    def _graph_distribution_noise(self, x_noise_range: np.ndarray, pdf: np.ndarray, index: int) -> None:
        """
        Plot the noise distribution and the theoretical Gaussian distribution.

        Args:
            x_noise_range (np.ndarray): The range of noise values.
            pdf (np.ndarray): The probability density function of the noise.
            index (int): The index of the subplot.
        """
        real_normal: np.ndarray = self._gaussian_function(x_noise_range)
        self.all_fig[1][1][index].plot(x_noise_range, pdf, color="cyan")
        self.all_fig[1][1][index].grid(visible=True, axis="both")
        self.all_fig[1][1][index].plot(x_noise_range, real_normal, color="black", linestyle="dotted")
    
    def _demo_noise_distribution(self, y_cum: np.ndarray, index: int) -> None:
        """
        Demonstrate the noise distribution by plotting the KDE and Gaussian functions.

        Args:
            y_cum (np.ndarray): The cumulative noisy outputs.
            index (int): The index of the subplot.
        """
        noise: np.ndarray = self.y_real - y_cum
        noise_min, noise_max = self._max_noise_range()
        x_noise_range: np.ndarray = np.arange(noise_min, noise_max, 0.01)
        
        pdf: np.ndarray = self._gaussian_kde_method(noise, x_noise_range)
        self._graph_distribution_noise(x_noise_range, pdf, index)
    
    def _graph_dist2d(self, x_cum: np.ndarray, y_cum: np.ndarray, index: int) -> None:
        """
        Plot a 2D histogram of the cumulative inputs and noisy outputs.

        Args:
            x_cum (np.ndarray): The cumulative inputs.
            y_cum (np.ndarray): The cumulative noisy outputs.
            index (int): The index of the subplot.
        """
        self.all_fig[1][0][index].hist2d(x_cum, y_cum, bins=self.bins, cmap="inferno")
    
    def demo(self, times: int) -> None:
        """
        Run the noise demonstration by applying the function with different noise levels and plotting the results.

        Args:
            times (int): The number of times to apply the function and add noise for cumulative outputs.
        """
        self._save_real_std_noise()
        
        for idx, mod in enumerate(self.std_mod):
            self._update_std_noise(mod)
            x_cum, y_cum = self.cumulative_func_noise(times)
            self._calc_y_real(x_cum) if idx == 0 else None
            self._demo_noise_distribution(y_cum, idx)
            self._graph_dist2d(x_cum, y_cum, idx)

        plt.show()
        plt.clf()