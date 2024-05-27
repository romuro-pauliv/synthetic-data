import numpy as np

def noise() -> np.float64:
    return np.random.uniform(np.random.uniform(-0.5, 0), np.random.uniform(0, 0.5))

def noise_sin(x: np.ndarray) -> np.ndarray:
    return np.array([np.sin(x_i) + noise() for x_i in x])

x   : np.ndarray = np.arange(-3, 3, 0.01)
y   : np.ndarray = np.sin(x)
y_n : np.ndarray = noise_sin(x)


def hist2d() -> tuple[np.ndarray]:
    x_cum: list[np.ndarray] = []
    y_cum: list[np.ndarray] = []
    for _ in range(100):
        x_cum.append(x)
        y_cum.append(noise_sin(x))

    return np.concatenate(x_cum), np.concatenate(y_cum)

x_cum, y_cum = hist2d()


# MODEL
from keras import Sequential
from keras.api.layers import Dense
from keras.api.layers import Input
from keras.api.losses import MeanSquaredError

model: Sequential = Sequential([
    Input((1, )),
    Dense(64, activation="sigmoid"),
    Dense(1, activation=None)
])

model.compile("adam", loss=MeanSquaredError)
model.fit(x_cum, y_cum, batch_size=32, epochs=25)

y_AI: np.ndarray = model.predict(x)
# |------------------------------------------------------|



import matplotlib.pyplot as plt
from matplotlib.axes    import Axes
from matplotlib.figure  import Figure

FIG: tuple[Figure, tuple[Axes]] = plt.subplots(1, 3)

FIG[1][0].hist2d(x_cum, y_cum, bins=300)
FIG[1][0].plot(x, y_AI, color="green")

FIG[1][1].plot(x, y, color="blue")
FIG[1][1].plot(x, y_AI, color="red", linestyle="dashed")

FIG[1][2].scatter(x, (y-np.concatenate(model.predict(x))), alpha=0.2)

plt.show()
