from array import array
import numpy as np


def step_function(value: float) -> int:
    if value >= 1:
        return 1
    return 0


def sigmoid(value: float) -> float:
    return 1 / (1+np.exp(-value))


def hyperbolic_tanget(value: float) -> float:
    return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value))


def relu(value: float) -> float:
    return np.max(0, value)


def softmax(value: array) -> array:
    ex = np.exp(value)
    return ex / ex.sum()


if __name__ == "__main__":
    print(sigmoid(2.1))
    print(hyperbolic_tanget(2.1))
    print(2.1)
