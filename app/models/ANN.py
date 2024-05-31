# +--------------------------------------------------------------------------------------------------------------------|
# |                                                                                                  app/models/ANN.py |
# |                                                                                          email: romulopauliv@bk.ru |
# |                                                                                                    encoding: UTF-8 |
# +--------------------------------------------------------------------------------------------------------------------|

# | Imports |----------------------------------------------------------------------------------------------------------|
from keras.api              import Sequential
from keras.api.layers       import Dense, Input, Conv1D, Reshape, MaxPooling1D, Flatten
from keras.api.losses       import MeanSquaredError
from keras.api.optimizers   import Adam

import numpy as np
# |--------------------------------------------------------------------------------------------------------------------|


class ANNReg(object):
    def __init__(self) -> None:
        self.learning_rate  : float     = 0.001

    def noised_data(self, x_cum: np.ndarray, y_cum: np.ndarray) -> None:
        self.x_cum: list[np.ndarray] = x_cum
        self.y_cum: list[np.ndarray] = y_cum

        self.batch_size     : int       = int(self.x_cum.shape[0]/1)
        self.epochs         : int       = 100
    
    def _model(self) -> None:
        self.model: Sequential = Sequential()
        
        self.model.add(Input(shape=(1,)))
        self.model.add(Dense(units=256, activation="relu", use_bias=True, kernel_initializer="normal"))
        self.model.add(Reshape(target_shape=(256, 1)))
        self.model.add(Conv1D(filters=64, kernel_size=(3), kernel_initializer="normal"))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=64, kernel_size=(3), kernel_initializer="normal"))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=64, kernel_size=(3), kernel_initializer="normal"))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation="relu", kernel_initializer="normal", use_bias=True))
        self.model.add(Dense(units=64, activation="relu", kernel_initializer="normal", use_bias=True))
        self.model.add(Dense(units=1, activation="linear", kernel_initializer="normal"))
        self.model.summary()
        input()
    
    def _compile(self) -> None:
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=MeanSquaredError)
    
    def _fit(self) -> None:
        self.model.fit(self.x_cum, self.y_cum, batch_size=self.batch_size, epochs=self.epochs, shuffle=True)
    
    def run(self) -> None:
        self._model()
        self._compile()
        self._fit()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(self.model.predict(x))