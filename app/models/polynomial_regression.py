# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                app/models/polynomial_regression.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
from sklearn.metrics import r2_score

import numpy as np
# |--------------------------------------------------------------------------------------------------------------------|


class PolyRegression(object):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the PolyRegression class with input and output data.

        Args:
            x (np.ndarray): The input data (independent variable).
            y (np.ndarray): The output data (dependent variable).
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.max_deg: int = 35
    
    def noised_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Store the cumulative input and output data with noise.

        Args:
            x (np.ndarray): The cumulative input data with noise.
            y (np.ndarray): The cumulative output data with noise.
        """
        self.x_cum: np.ndarray = x
        self.y_cum: np.ndarray = y
    
    def poly_optimizer(self) -> tuple[np.ndarray, float, int]:
        """
        Find the best polynomial degree for fitting the noisy data using R2 score as the optimization criterion.

        Returns:
            tuple[np.ndarray, float, int]: The best-fit polynomial output, the best R2 score, and the degree of the 
                                           polynomial.
        """
        self.best_r2_score: float = -1
        self.deg: int = 0
        self.best_y_poly: np.ndarray = np.array([])
        
        for i in range(1, self.max_deg):
            poly_func = np.poly1d(np.polyfit(self.x_cum, self.y_cum, i))
            y_poly: np.ndarray = poly_func(self.x)
            r2_score_result: float = r2_score(self.y, y_poly)
            
            if r2_score_result >= self.best_r2_score:
                self.best_r2_score: float = r2_score_result
                self.deg: int = i
                self.best_y_poly: np.ndarray = poly_func
            
            print("\rBest R2:", round(self.best_r2_score, 4), "deg:", i, end="")
        print("")
            
        return self.best_y_poly, self.best_r2_score, self.deg

    def best_model(self, x: np.ndarray) -> np.ndarray:
        return self.best_y_poly(x)