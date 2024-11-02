import numpy as np

F1 = lambda x: -0.005 * (x - 0.5) * (x - 5) * (x - 8) * (x - 13.4)


def F2(x):
    a = 1
    b = 0.7
    c = 0.4 * np.pi
    return (
        -a * np.exp(-b * np.sqrt(0.5 * x**2))
        - np.exp(0.5 * (np.cos(c * x)))
        + a
        + np.exp(1.0)
        - 0.7
    )
