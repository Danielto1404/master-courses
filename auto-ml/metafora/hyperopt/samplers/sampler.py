import random

import numpy as np


def simulated_annealing(
        iterations,
        start,
        crossover,
        transform,
        criterion
):
    x = crossover(start)
    best = criterion(transform(x))

    for i in range(iterations):
        n_x = crossover(x)
        score = criterion(transform(x))

        gain = score - best
        discount = np.sqrt(i + 1)

        if random.random() < np.exp(gain / discount):
            best = score
            x = n_x

    return x


__all__ = [
    "simulated_annealing"
]
