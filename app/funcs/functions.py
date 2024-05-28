# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                             app/funcs/functions.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
import numpy as np
# |--------------------------------------------------------------------------------------------------------------------|


class Functions:
    @staticmethod
    def gaussian_distribution(x: np.ndarray, std: float) -> np.ndarray:
        return (1/2*np.pi*np.sqrt(std))*np.e**(-((x)**2)/(2*std))
    
    @staticmethod
    def modulated_normal(x: np.ndarray) -> np.ndarray:
        return np.e**(-x**2)*np.cos(2*np.pi*x)
    
    @staticmethod
    def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return (x*a + b)
    
    @staticmethod
    def test(x: np.ndarray, a: float) -> np.ndarray:
        return np.sin(x)/(a*x)
    