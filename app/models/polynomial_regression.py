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
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.max_deg: int = 35
    
    def noised_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_cum: np.ndarray = x
        self.y_cum: np.ndarray = y
    
    def poly_optimizer(self) -> None:
        self.best_r2_score: float = 0
        self.deg: int = 0
        self.best_y_poly: np.ndarray = 0
        
        for i in range(1, self.max_deg):
            y_poly: np.ndarray = np.poly1d(np.polyfit(self.x_cum, self.y_cum, i))(self.x)
            r2_score_result: float = r2_score(self.y, y_poly)
            
            if r2_score_result >= self.best_r2_score:
                self.best_r2_score: float = r2_score_result
                self.deg: int = i
                self.best_y_poly: np.ndarray = y_poly
            
            print("\rBest R2:", round(self.best_r2_score, 4), "deg:", i, end="")
        print("")
            
        return self.best_y_poly, self.best_r2_score, self.deg
    