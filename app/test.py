import numpy as np

from keras.api          import Sequential
from keras.api.layers   import Dense, Input
from keras.api.losses   import MeanSquaredError
from keras.api.optimizers import Adam

# CONFIG |-------------------------------------------|
xmin: float = -np.pi
xmax: float = np.pi
step: float = 0.01

minnoise: float = -4
maxnoise: float = 4

std         : float = 0.5
noise_times : int = 10

batch_size  : int = 32
epochs      : int = 1000
# |--------------------------------------------------|

def func(x: np.ndarray, std: np.float64) -> np.ndarray:
#    return (1/2*np.pi*np.sqrt(std))*np.e**(-((x-0)**2)/(2*std))
    return np.e**(-x**2)*np.cos(2*np.pi*x)
#    return x

def noise() -> np.float64:
    return np.random.uniform(np.random.uniform(minnoise, 0), np.random.uniform(maxnoise, 1))

def noise_func(x: np.ndarray, std: np.float64) -> np.ndarray:
    return np.array([func(x_i, std) + noise() for x_i in x])

def noise_data(x: np.ndarray, std: np.float64, times: int) -> tuple[np.ndarray]:
    x_cum: list[np.ndarray] = []
    y_cum: list[np.ndarray] = []
    for _ in range(times):
        x_cum.append(x)
        y_cum.append(noise_func(x, std))
    return np.concatenate(x_cum), np.concatenate(y_cum)

def Model(x: np.ndarray, y: np.ndarray) -> Sequential:
    model: Sequential = Sequential([
        Input((1, )),
        Dense(64, activation='relu'),
        Dense(64, activation='sigmoid'),
        Dense(1, activation="linear")
    ])
    model.compile(Adam(learning_rate=0.0005), loss=MeanSquaredError)
    model.fit(x, y, batch_size=batch_size, epochs=epochs)
    return model


# | (X, Y) |
x: np.ndarray = np.arange(xmin, xmax, step)
y: np.ndarray = func(x, std)

# | Noise (X, Y) |
X_C, Y_C = noise_data(x, std, noise_times)

# | Model |
model_regression: Sequential = Model(X_C, Y_C)

# Graphs
import matplotlib.pyplot as plt
from matplotlib.axes    import Axes
from matplotlib.figure  import Figure

fig: tuple[Figure, tuple[Axes]] = plt.subplots(1, 3)

fig[1][0].hist2d(X_C, Y_C, bins=300)

fig[1][1].plot(x, y, color="blue", label="real function")
fig[1][1].plot(x, model_regression.predict(x), color="red", linestyle="dashed")

plt.show()