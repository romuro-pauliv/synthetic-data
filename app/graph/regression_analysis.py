# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                   app/graph/regression_analysis.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
import matplotlib.pyplot as plt

from matplotlib.figure  import Figure
from matplotlib.axes    import Axes

from scipy.stats import gaussian_kde

import numpy as np

from typing import Callable, Any
# |--------------------------------------------------------------------------------------------------------------------|


class RegressionAnalysis(object):
    def __init__(self) -> None:
        self._fig()
        self.bins: int = 300
    
    def _fig(self) -> None:
        self.all_fig: tuple[Figure, tuple[Axes]] = plt.subplots(1, 3, figsize=(16, 4))

    # | Class Inputs |-------------------------------------------------------------------------------------------------|
    
    def define_noised_data(self, noised_x_cum: np.ndarray, noised_y_cum: np.ndarray) -> None:
        self.noised_x_cum: np.ndarray = noised_x_cum
        self.noised_y_cum: np.ndarray = noised_y_cum
    
    def define_function(self, func: Callable[[np.ndarray, Any], np.ndarray]) -> None:
        self.func: Callable[[np.ndarray, Any], np.ndarray] = func
    
    def define_inputs(self, *args) -> None:
        self.inputs: tuple[np.ndarray, Any] = args
    
    def define_noise_range(self, std: float) -> None:
        self.n_std: float = std
    
    def define_regression_model(self, reg_model: Callable[[np.ndarray], np.ndarray]) -> None:
        self.reg_model: Callable[[np.ndarray], np.ndarray] = reg_model
    
    # |----------------------------------------------------------------------------------------------------------------|
    
    # | Analysis |-----------------------------------------------------------------------------------------------------|
    def _get_regression(self) -> None:
        self.x      : np.ndarray = self.inputs[0]
        self.y_reg  : np.ndarray = self.reg_model(self.x)
    
    def _get_real_y(self) -> None:
        self.y_real : np.ndarray = self.func(*self.inputs)
    
    def _run_analysis(self) -> None:
        self._get_regression()
        self._get_real_y()
    # |----------------------------------------------------------------------------------------------------------------|
    
    # | Graphs |-------------------------------------------------------------------------------------------------------|
    
    def graph_regression_with_noise(self) -> None:
        self.all_fig[1][0].hist2d(self.noised_x_cum, self.noised_y_cum, bins=self.bins)
        self.all_fig[1][0].plot(self.x, self.y_reg, color="white", linestyle="dotted")
    
    def graph_comparison_y_y_reg(self) -> None:
        self.all_fig[1][1].plot(self.x, self.y_reg, color="black", linestyle="dotted")
        self.all_fig[1][1].plot(self.x, self.y_real, color="gray")
        self.all_fig[1][1].grid(True, "both")
    
    def graph_error(self) -> None:
        diff: np.ndarray = (self.y_reg - self.y_real)
        zeros: np.ndarray = [0]*self.x.shape[0]
        
        self.all_fig[1][2].scatter(self.x, diff, s=0.1, color="black")
        self.all_fig[1][2].plot(self.x, zeros, linestyle="dashed", color="black")
        self.all_fig[1][2].fill_between(self.x, zeros, diff, alpha=0.2, color="black")
        self.all_fig[1][2].grid(True, "both")
        
    # |----------------------------------------------------------------------------------------------------------------|
    
    def run(self) -> None:
        self._run_analysis()
        
        self.graph_regression_with_noise()
        self.graph_comparison_y_y_reg()
        self.graph_error()
        
        plt.show()
        plt.clf()
    # |----------------------------------------------------------------------------------------------------------------|
    